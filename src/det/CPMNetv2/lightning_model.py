import math
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist

PACKAGE_ROOT = Path(__file__).resolve().parent
import sys

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from networks.ResNet_3D_CPM import Detection_Postprocess, Detection_loss, resnet18
from optimizer.optim import AdamW
from optimizer.scheduler import GradualWarmupScheduler
from evaluationScript.detectionCADEvalutionIOU import noduleCADEvaluation
from utils.box_utils import nms_3D


class CPMNetv2LitModel(pl.LightningModule):
    """PyTorch Lightning wrapper around the original CPMNetv2 detector."""

    def __init__(
        self,
        crop_size: Sequence[int] = (64, 128, 128),
        spacing: Sequence[float] = (0.7, 0.3125, 0.3125),
        n_channels: int = 1,
        n_blocks: Sequence[int] = (2, 3, 3, 3),
        n_filters: Sequence[int] = (64, 96, 128, 160),
        stem_filters: int = 32,
        norm_type: str = "batchnorm",
        head_norm: str = "batchnorm",
        act_type: str = "ReLU",
        se: bool = False,
        topk: int = 5,
        lambda_cls: float = 4.0,
        lambda_shape: float = 0.1,
        lambda_offset: float = 1.0,
        lambda_iou: float = 1.0,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        warmup_multiplier: float = 10.0,
        warmup_epochs: int = 2,
        cosine_t_max: int = 300,
        eta_min: float = 1e-6,
        post_topk: int = 60,
        post_threshold: float = 0.15,
        post_nms_threshold: float = 0.05,
        post_num_topk: int = 20,
        final_nms_overlap: float = 0.05,
        final_topk: int = 40,
        inference_batch_multiplier: int = 2,
        prediction_dir: str = "predictions",
        evaluate_froc: bool = True,
        froc_iou_threshold: float = 0.1,
        annotations_excluded_csv: str = str(PACKAGE_ROOT / "evaluationScript" / "annotations_excluded.csv"),
        confidence_log_interval: int = 50,
        debug_target_stats: bool = False,
        debug_target_stats_interval: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.crop_size = list(crop_size)
        self.spacing = list(spacing)
        self.lambda_cls = lambda_cls
        self.lambda_shape = lambda_shape
        self.lambda_offset = lambda_offset
        self.lambda_iou = lambda_iou
        self.final_nms_overlap = final_nms_overlap
        self.final_topk = final_topk
        self.inference_batch_multiplier = inference_batch_multiplier
        self.prediction_dir = prediction_dir
        self.evaluate_froc = evaluate_froc
        self.froc_iou_threshold = froc_iou_threshold
        self.annotations_excluded_csv = annotations_excluded_csv
        self.confidence_log_interval = int(confidence_log_interval)
        self.validation_predictions: List[list] = []
        self.test_predictions: List[list] = []

        detection_loss = Detection_loss(
            crop_size=self.crop_size,
            topk=topk,
            spacing=self.spacing,
            debug_target_stats=debug_target_stats,
            debug_target_stats_interval=debug_target_stats_interval,
        )
        self.model = resnet18(
            n_channels=n_channels,
            n_blocks=list(n_blocks),
            n_filters=list(n_filters),
            stem_filters=stem_filters,
            norm_type=norm_type,
            head_norm=head_norm,
            act_type=act_type,
            se=se,
            first_stride=(1, 2, 2),
            detection_loss=detection_loss,
            device=torch.device("cpu"),
        )
        self.postprocess = Detection_Postprocess(
            topk=post_topk,
            threshold=post_threshold,
            nms_threshold=post_nms_threshold,
            num_topk=post_num_topk,
            crop_size=self.crop_size,
        )

    def forward(self, image):
        return self.model(image)

    def _sync_model_device(self):
        self.model.device = self.device

    def _loss(self, batch):
        self._sync_model_device()
        data = batch["image"].to(self.device)
        labels = batch["annot"].to(self.device)
        if self.model.training:
            cls_loss, shape_loss, offset_loss, iou_loss = self.model([data, labels])
        else:
            output = self.model(data)
            cls_loss, shape_loss, offset_loss, iou_loss = self.model.detection_loss(output, labels, device=self.device)
        cls_loss = cls_loss.mean()
        shape_loss = shape_loss.mean()
        offset_loss = offset_loss.mean()
        iou_loss = iou_loss.mean()
        loss = (
            self.lambda_cls * cls_loss
            + self.lambda_shape * shape_loss
            + self.lambda_offset * offset_loss
            + self.lambda_iou * iou_loss
        )
        return loss, cls_loss, shape_loss, offset_loss, iou_loss

    @staticmethod
    def _cls_confidence_stats(output):
        probs = torch.sigmoid(output["Cls"].detach()).flatten().float()
        topk_n = min(20, probs.numel())
        topk = torch.topk(probs, k=topk_n, largest=True).values
        return {
            "conf_mean": probs.mean(),
            "conf_p95": torch.quantile(probs, 0.95),
            "conf_p99": torch.quantile(probs, 0.99),
            "conf_max": probs.max(),
            "conf_top20_mean": topk.mean(),
            "conf_frac_gt_0_01": (probs > 0.01).float().mean(),
            "conf_frac_gt_0_05": (probs > 0.05).float().mean(),
            "conf_frac_gt_post_threshold": None,
        }

    def _log_confidence_distribution(self, batch, stage: str, on_step: bool, on_epoch: bool):
        if stage == "train" and self.confidence_log_interval <= 0:
            return
        if stage == "train" and self.global_step % self.confidence_log_interval != 0:
            return

        was_training = self.model.training
        self.model.eval()
        self._sync_model_device()
        with torch.no_grad():
            output = self.model(batch["image"].to(self.device))
        if was_training:
            self.model.train()

        stats = self._cls_confidence_stats(output)
        post_threshold = float(self.hparams.post_threshold)
        probs = torch.sigmoid(output["Cls"].detach()).flatten().float()
        stats["conf_frac_gt_post_threshold"] = (probs > post_threshold).float().mean()

        batch_size = batch["image"].shape[0]
        for name, value in stats.items():
            self.log(
                f"{stage}/{name}",
                value,
                prog_bar=name == "conf_max",
                on_step=on_step,
                on_epoch=on_epoch,
                batch_size=batch_size,
            )

    def training_step(self, batch, batch_idx):
        loss, cls_loss, shape_loss, offset_loss, iou_loss = self._loss(batch)
        if not torch.isfinite(loss):
            self.log("train/nonfinite_loss_batches", 1.0, prog_bar=True, on_step=True, on_epoch=True)
            print(f"Skipping non-finite training loss at epoch={self.current_epoch}, batch_idx={batch_idx}")
            return None
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch["image"].shape[0])
        self.log("train/cls_loss", cls_loss, on_step=False, on_epoch=True)
        self.log("train/shape_loss", shape_loss, on_step=False, on_epoch=True)
        self.log("train/offset_loss", offset_loss, on_step=False, on_epoch=True)
        self.log("train/iou_loss", iou_loss, on_step=False, on_epoch=True)
        self._log_confidence_distribution(batch, "train", on_step=True, on_epoch=False)
        return loss

    def _nzhw(self, sample):
        nzhw = sample["nzhw"]
        if isinstance(nzhw, torch.Tensor):
            return [int(v) for v in nzhw.flatten().tolist()]
        return [int(v.item()) if torch.is_tensor(v) else int(v) for v in nzhw]

    def _spacing(self, sample):
        spacing = sample["spacing"]
        if isinstance(spacing, torch.Tensor):
            return spacing.detach().cpu().numpy().reshape(-1)[:3]
        values = []
        for value in spacing:
            if torch.is_tensor(value):
                values.append(float(value.detach().cpu().reshape(-1)[0].item()))
            else:
                values.append(float(np.asarray(value).reshape(-1)[0]))
        return np.asarray(values)

    def _predict_sample(self, sample):
        self.model.eval()
        self._sync_model_device()
        data = sample["split_images"][0].to(self.device)
        nzhw = self._nzhw(sample)
        datamodule = self.trainer.datamodule if self.trainer else None
        loader_batch_size = int(getattr(datamodule, "batch_size", 1))
        num_samples = int(getattr(datamodule, "num_samples", 1))
        batch_size = max(1, self.inference_batch_multiplier * loader_batch_size * num_samples)
        outputlist = []

        for start in range(0, data.size(0), batch_size):
            stop = min(start + batch_size, data.size(0))
            with torch.no_grad():
                output = self.model(data[start:stop])
                output = self.postprocess(output, device=self.device)
            outputlist.append(output.detach().cpu().numpy())

        output = np.concatenate(outputlist, axis=0)
        splitcomb = self._prediction_splitcomb()
        if splitcomb is None:
            raise RuntimeError("CPMNetv2 prediction requires a datamodule with DetDatasetCSVRTest.splitcomb.")
        output = splitcomb.combine(output, nzhw=nzhw)
        output = torch.from_numpy(output).view(-1, 8)
        output = output[output[:, 0] != -1.0]
        if len(output) > 0:
            keep = nms_3D(output[:, 1:], overlap=self.final_nms_overlap, top_k=self.final_topk)
            output = output[keep]
        return output.detach().cpu().numpy()

    def _prediction_splitcomb(self):
        datamodule = self.trainer.datamodule if self.trainer else None
        if datamodule is None:
            return None

        stage = getattr(getattr(self.trainer, "state", None), "stage", None)
        stage_name = str(stage).lower() if stage is not None else ""
        if "test" in stage_name:
            dataset_names = ("test_ds", "test_dataset", "val_ds", "val_dataset")
        elif "valid" in stage_name or "sanity" in stage_name:
            dataset_names = ("val_ds", "val_dataset", "test_ds", "test_dataset")
        else:
            dataset_names = ("test_ds", "test_dataset", "val_ds", "val_dataset")

        for name in dataset_names:
            dataset = getattr(datamodule, name, None)
            splitcomb = getattr(dataset, "splitcomb", None)
            if splitcomb is not None:
                return splitcomb
        return None

    def _rows_from_output(self, sample, output):
        name = sample["file_name"][0]
        rows = []
        for det in output:
            rows.append([name, det[4], det[3], det[2], det[1], det[7], det[6], det[5]])
        return rows

    def validation_step(self, batch, batch_idx):
        if "annot" not in batch:
            output = self._predict_sample(batch)
            self.validation_predictions.extend(self._rows_from_output(batch, output))
            self.log("val/num_detections", float(len(output)), on_step=False, on_epoch=True, prog_bar=True)
            return None
        loss, cls_loss, shape_loss, offset_loss, iou_loss = self._loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch["image"].shape[0])
        self.log("val/cls_loss", cls_loss, on_step=False, on_epoch=True, batch_size=batch["image"].shape[0])
        self.log("val/shape_loss", shape_loss, on_step=False, on_epoch=True, batch_size=batch["image"].shape[0])
        self.log("val/offset_loss", offset_loss, on_step=False, on_epoch=True, batch_size=batch["image"].shape[0])
        self.log("val/iou_loss", iou_loss, on_step=False, on_epoch=True, batch_size=batch["image"].shape[0])
        self._log_confidence_distribution(batch, "val", on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        output = self._predict_sample(batch)
        self.test_predictions.extend(self._rows_from_output(batch, output))
        self.log("test/num_detections", float(len(output)), on_step=False, on_epoch=True, prog_bar=True)

    def _write_predictions(self, rows, stage: str):
        out_dir = Path(self.trainer.default_root_dir) / self.prediction_dir
        if self.trainer.is_global_zero:
            out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"epoch_{self.current_epoch:03d}" if stage == "val" else "test"
        path = out_dir / f"{stage}_predictions_{suffix}.csv"
        columns = ["seriesuid", "coordX", "coordY", "coordZ", "probability", "w", "h", "d"]
        if self.trainer.is_global_zero:
            pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
        return path

    def _write_ground_truth_files(self, stage: str):
        datamodule = self.trainer.datamodule if self.trainer else None
        dataset = None
        if datamodule is not None:
            dataset = getattr(datamodule, "val_ds", None) if stage == "val" else getattr(datamodule, "test_ds", None)
        if dataset is None or not hasattr(dataset, "annotations_dataframe"):
            return None, None

        out_dir = Path(self.trainer.default_root_dir) / self.prediction_dir / f"{stage}_gt"
        if not self.trainer.is_global_zero:
            return None, None
        out_dir.mkdir(parents=True, exist_ok=True)
        annotation_df, seriesuid_df = dataset.annotations_dataframe()
        annotation_path = out_dir / f"annotation_{stage}.csv"
        seriesuid_path = out_dir / f"seriesuid_{stage}.csv"
        annotation_df.to_csv(annotation_path, index=False)
        seriesuid_df.to_csv(seriesuid_path, index=False, header=False)
        return annotation_path, seriesuid_path

    def _evaluate_predictions(self, prediction_path, stage: str):
        if not self.evaluate_froc:
            return
        payload = {"ok": False, "mean_froc": 0.0, "frocs": [0.0] * 7}
        if self.trainer.is_global_zero:
            annotation_path, seriesuid_path = self._write_ground_truth_files(stage)
            if annotation_path is not None:
                output_dir = Path(self.trainer.default_root_dir) / self.prediction_dir / f"{stage}_froc_epoch_{self.current_epoch:03d}"
                output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    result = noduleCADEvaluation(
                        str(annotation_path),
                        self.annotations_excluded_csv,
                        str(seriesuid_path),
                        str(prediction_path),
                        str(output_dir),
                        self.froc_iou_threshold,
                    )
                    frocs_raw = [float(v) for v in result[-1]]
                    if any(not np.isfinite(v) for v in frocs_raw):
                        print(f"{stage} FROC contained non-finite values; replacing them with 0.0")
                    frocs = [0.0 if not np.isfinite(v) else float(v) for v in frocs_raw]
                    payload = {"ok": True, "mean_froc": float(np.mean(np.asarray(frocs))), "frocs": frocs}
                    print(f"{stage} FROC saved to {output_dir}; mean={payload['mean_froc']:.4f}")
                except Exception as exc:
                    print(f"{stage} FROC compute error: {exc}")

        if dist.is_available() and dist.is_initialized():
            objects = [payload]
            dist.broadcast_object_list(objects, src=0)
            payload = objects[0]
        if not payload["ok"]:
            return

        mean_froc = float(payload["mean_froc"])
        frocs = payload["frocs"]
        self.log(f"{stage}/mean_froc", mean_froc, prog_bar=True, on_step=False, on_epoch=True)
        if stage == "val":
            self.log("val_mean_froc", mean_froc, logger=False, on_step=False, on_epoch=True)
            mean_tensor = torch.tensor(mean_froc, device=self.device)
            self.trainer.callback_metrics["val/mean_froc"] = mean_tensor
            self.trainer.callback_metrics["val_mean_froc"] = mean_tensor
        for fp, score in zip([0.125, 0.25, 0.5, 1, 2, 4, 8], frocs):
            self.log(f"{stage}/froc_{fp:g}fp", float(score), on_step=False, on_epoch=True)

    def on_validation_epoch_start(self):
        self.validation_predictions = []

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self.validation_predictions:
            if not self.trainer.sanity_checking:
                self.log("val/mean_froc", 0.0, prog_bar=True, on_step=False, on_epoch=True)
                self.log("val_mean_froc", 0.0, logger=False, on_step=False, on_epoch=True)
                mean_tensor = torch.tensor(0.0, device=self.device)
                self.trainer.callback_metrics["val/mean_froc"] = mean_tensor
                self.trainer.callback_metrics["val_mean_froc"] = mean_tensor
            return
        rows = self._gather_prediction_rows(self.validation_predictions)
        path = self._write_predictions(rows, "val")
        self._evaluate_predictions(path, "val")

    def on_test_epoch_start(self):
        self.test_predictions = []

    def on_test_epoch_end(self):
        rows = self._gather_prediction_rows(self.test_predictions)
        path = self._write_predictions(rows, "test")
        if self.trainer.is_global_zero:
            self.log("test/prediction_file_written", 1.0, logger=False)
            print(f"Test predictions saved to {path}")
        self._evaluate_predictions(path, "test")

    @staticmethod
    def _gather_prediction_rows(rows):
        if not (dist.is_available() and dist.is_initialized()):
            return rows
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, rows)
        merged = []
        for rank_rows in gathered:
            merged.extend(rank_rows)
        return merged

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.cosine_t_max,
            eta_min=self.hparams.eta_min,
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=self.hparams.warmup_multiplier,
            total_epoch=self.hparams.warmup_epochs,
            after_scheduler=cosine,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def lr_scheduler_step(self, scheduler, metric):
        try:
            scheduler.step(metrics=metric)
        except TypeError:
            scheduler.step()
