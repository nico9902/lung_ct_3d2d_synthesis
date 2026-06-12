from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.det.SCPMNet.dataset import SCPMSlidingWindowDataset, scpm_sliding_collate
from src.det.SCPMNet.lightning_model import SCPMLitModel, sphere_nms


class SlidingWindowValidationFROCCallback(pl.Callback):
    def __init__(
        self,
        every_n_epochs: int = 10,
        batch_size: int = 1,
        num_workers: int = 0,
        start_epoch: int = 1,
        output_subdir: str = "validation_froc",
    ):
        super().__init__()
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.batch_size = max(1, int(batch_size))
        self.num_workers = max(0, int(num_workers))
        self.start_epoch = max(1, int(start_epoch))
        self.output_subdir = output_subdir
        self.dataset: SCPMSlidingWindowDataset | None = None
        self.gt_by_series: dict[str, np.ndarray] = {}

    def setup(self, trainer: pl.Trainer, pl_module: SCPMLitModel, stage: str) -> None:
        if stage != "fit":
            return
        datamodule = trainer.datamodule
        if datamodule is None:
            raise ValueError("Sliding-window validation FROC requires a datamodule.")
        hparams = datamodule.hparams
        self.dataset = SCPMSlidingWindowDataset(
            csv_path=hparams.csv_path,
            split="val",
            data_root=hparams.data_root,
            crop_size=tuple(hparams.crop_size),
            stride=tuple(hparams.sliding_window_stride),
            clip=tuple(hparams.clip),
            skip_missing_images=hparams.skip_missing_images,
        )
        self.gt_by_series = {str(seriesuid): pl_module._gt_from_rows(rows) for seriesuid, rows in self.dataset.groups}
        if len(self.dataset) == 0 or not self.gt_by_series:
            raise ValueError("Sliding-window validation FROC found no validation windows/series.")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: SCPMLitModel) -> None:
        epoch = int(trainer.current_epoch) + 1
        if trainer.sanity_checking or epoch < self.start_epoch or epoch % self.every_n_epochs != 0:
            return
        if self.dataset is None:
            return

        was_training = pl_module.training
        pl_module.eval()
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=scpm_sliding_collate,
            pin_memory=getattr(trainer.datamodule.hparams, "pin_memory", False) if trainer.datamodule else False,
        )

        rows: list[list[float | str]] = []
        columns = ["seriesuid", "coordZ", "coordY", "coordX", "radius", "probability"]
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(pl_module.device)
                outputs = pl_module(images)
                origins = batch["origin"].cpu()
                for i, seriesuid in enumerate(batch["seriesuid"]):
                    detections = pl_module.decode_one(outputs, i).detach().cpu()
                    if len(detections):
                        detections[:, :3] += origins[i].view(1, 3)
                    for z, y, x, radius, score in detections.tolist():
                        rows.append([seriesuid, z, y, x, radius, score])

        final_rows: list[list[float | str]] = []
        if rows:
            pred_df = pd.DataFrame(rows, columns=columns)
            for seriesuid, group in pred_df.groupby("seriesuid", sort=False):
                detections = torch.as_tensor(
                    group[["coordZ", "coordY", "coordX", "radius", "probability"]].to_numpy(),
                    dtype=torch.float32,
                )
                detections = sphere_nms(detections, pl_module.hparams.nms_threshold, pl_module.hparams.final_topk)
                for z, y, x, radius, score in detections.tolist():
                    final_rows.append([seriesuid, z, y, x, radius, score])

        final_df = pd.DataFrame(final_rows, columns=columns)
        froc, curve, mean_froc = self._evaluate(final_df, pl_module)
        out_dir = Path(trainer.default_root_dir) / self.output_subdir / f"epoch_{epoch:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_dir / "val_predictions.csv", index=False)
        froc.to_csv(out_dir / "val_froc.csv", index=False)
        curve.to_csv(out_dir / "val_froc_curve.csv", index=False)

        mean_tensor = torch.tensor(mean_froc, device=pl_module.device)
        pl_module.log("val/mean_froc", mean_tensor, prog_bar=True, logger=True, sync_dist=False)
        pl_module.log("val_mean_froc", mean_tensor, logger=False, sync_dist=False)
        trainer.callback_metrics["val/mean_froc"] = mean_tensor
        trainer.callback_metrics["val_mean_froc"] = mean_tensor
        for _, row in froc.iterrows():
            metric = f"val/froc_{row.fp_per_scan:g}fp"
            value = torch.tensor(float(row.sensitivity), device=pl_module.device)
            pl_module.log(metric, value, logger=True, sync_dist=False)
            trainer.callback_metrics[metric] = value

        if was_training:
            pl_module.train()

    def _evaluate(self, pred_df: pd.DataFrame, pl_module: SCPMLitModel) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        total_gt = int(sum(len(v) for v in self.gt_by_series.values()))
        num_scans = int(len(self.gt_by_series))
        if total_gt == 0 or num_scans == 0:
            raise ValueError("Cannot evaluate validation FROC without ground-truth nodules.")

        if pred_df.empty:
            fp_per_scan = np.asarray([0.0], dtype=np.float32)
            sensitivity = np.asarray([0.0], dtype=np.float32)
        else:
            pred_df = pred_df.sort_values("probability", ascending=False).reset_index(drop=True)
            matched = {seriesuid: np.zeros(len(gt), dtype=bool) for seriesuid, gt in self.gt_by_series.items()}
            tp = 0
            fp = 0
            sensitivities = [0.0]
            fps = [0.0]
            for _, row in pred_df.iterrows():
                seriesuid = str(row["seriesuid"])
                gt = self.gt_by_series.get(seriesuid, np.zeros((0, 4), dtype=np.float32))
                pred = row[["coordZ", "coordY", "coordX", "radius"]].to_numpy(dtype=np.float32)
                match_idx = pl_module._match_prediction(pred, gt, matched.get(seriesuid, np.zeros(0, dtype=bool)))
                if match_idx is None:
                    fp += 1
                else:
                    matched[seriesuid][match_idx] = True
                    tp += 1
                sensitivities.append(tp / total_gt)
                fps.append(fp / num_scans)
            fp_per_scan = np.asarray(fps, dtype=np.float32)
            sensitivity = np.maximum.accumulate(np.asarray(sensitivities, dtype=np.float32))

        rows = []
        values = []
        for rate in tuple(float(v) for v in pl_module.hparams.froc_fp_rates):
            eligible = sensitivity[fp_per_scan <= rate]
            score = float(eligible.max()) if eligible.size else 0.0
            rows.append({"fp_per_scan": rate, "sensitivity": score})
            values.append(score)
        return pd.DataFrame(rows), pd.DataFrame({"fp_per_scan": fp_per_scan, "sensitivity": sensitivity}), float(np.mean(values))
