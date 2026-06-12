from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist

from src.det.SCPMNet.losses import SCPMDetectionLoss
from src.det.SCPMNet.model import scpmnet18


def sphere_nms(detections: torch.Tensor, iou_threshold: float, topk: int) -> torch.Tensor:
    if detections.numel() == 0:
        return detections
    detections = detections[detections[:, 4].argsort(descending=True)]
    keep = []
    while detections.numel() and len(keep) < topk:
        current = detections[0]
        keep.append(current)
        if detections.size(0) == 1:
            break
        rest = detections[1:]
        dist = torch.linalg.vector_norm(rest[:, :3] - current[:3], dim=1)
        r1 = current[3].clamp_min(1e-6)
        r2 = rest[:, 3].clamp_min(1e-6)
        no_overlap = dist >= (r1 + r2)
        contained = (dist + torch.minimum(r1, r2)) <= torch.maximum(r1, r2)
        inter_vol = torch.zeros_like(dist)
        inter_vol = torch.where(contained, (4.0 / 3.0) * torch.pi * torch.minimum(r1, r2).pow(3), inter_vol)
        partial = ~(no_overlap | contained)
        if partial.any():
            d = dist[partial].clamp_min(1e-6)
            rp = r2[partial]
            cos1 = ((r1.square() + d.square() - rp.square()) / (2 * r1 * d)).clamp(-0.999999, 0.999999)
            cos2 = ((rp.square() + d.square() - r1.square()) / (2 * rp * d)).clamp(-0.999999, 0.999999)
            h1 = r1 * (1 - cos1)
            h2 = rp * (1 - cos2)
            inter_vol[partial] = torch.pi * h1.square() * (r1 - h1 / 3) + torch.pi * h2.square() * (rp - h2 / 3)
        union = (4.0 / 3.0) * torch.pi * (r1.pow(3) + r2.pow(3)) - inter_vol
        iou = inter_vol / union.clamp_min(1e-6)
        detections = rest[iou <= iou_threshold]
    return torch.stack(keep) if keep else detections.new_zeros((0, detections.size(1)))


class SCPMLitModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        stride: int = 2,
        positive_topk: int = 7,
        positive_radius_factor: float = 0.6,
        neg_pos_ratio: int = 20,
        focal_gamma: float = 2.0,
        refocal_threshold: float = 0.9,
        refocal_weight: float = 4.0,
        smooth_l1_beta: float = 1.0 / 9.0,
        lambda_cls: float = 1.0,
        lambda_radius: float = 1.0,
        lambda_offset: float = 1.0,
        lambda_siou: float = 1.0,
        using_sac: bool = False,
        optimizer_name: str = "adamw",
        momentum: float = 0.9,
        warmup_epochs: int = 0,
        warmup_lr: float = 1e-4,
        lr_milestones: tuple[int, ...] = (80, 150),
        lr_gamma: float = 0.1,
        decode_threshold: float = 0.2,
        decode_topk: int = 100,
        nms_threshold: float = 0.05,
        final_topk: int = 100,
        evaluate_froc: bool = True,
        evaluate_val_froc: bool | None = None,
        evaluate_test_froc: bool | None = None,
        val_loader_names: tuple[str, ...] = ("val",),
        froc_match_strategy: str = "center_distance",
        froc_iou_threshold: float = 0.1,
        froc_fp_rates: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        prediction_dir: str = "predictions",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = scpmnet18(using_sac=using_sac)
        self.loss_fn = SCPMDetectionLoss(
            stride=stride,
            positive_topk=positive_topk,
            positive_radius_factor=positive_radius_factor,
            neg_pos_ratio=neg_pos_ratio,
            focal_gamma=focal_gamma,
            refocal_threshold=refocal_threshold,
            refocal_weight=refocal_weight,
            smooth_l1_beta=smooth_l1_beta,
            lambda_cls=lambda_cls,
            lambda_radius=lambda_radius,
            lambda_offset=lambda_offset,
            lambda_siou=lambda_siou,
        )
        self.test_rows: list[list[float | str]] = []
        self.val_rows: list[list[float | str]] = []

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(image)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        outputs = self(batch["image"])
        losses = self.loss_fn(outputs, batch["annot"].to(self.device))
        batch_size = batch["image"].size(0)
        for name, value in losses.items():
            self.log(
                f"{stage}/{name}",
                value,
                prog_bar=name == "loss",
                on_step=stage == "train",
                on_epoch=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
        return losses["loss"]

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def _val_stage_name(self, dataloader_idx: int) -> str:
        names = tuple(str(name) for name in self.hparams.val_loader_names)
        if len(names) <= 1 and names[0] == "val":
            return "val"
        name = names[dataloader_idx] if dataloader_idx < len(names) else f"loader_{dataloader_idx}"
        return f"val/{name}"

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        stage = self._val_stage_name(dataloader_idx)
        if "annot" not in batch:
            outputs = self(batch["image"])
            for i, seriesuid in enumerate(batch["seriesuid"]):
                detections = self.decode_one(outputs, i).detach().cpu()
                if "origin" in batch and len(detections):
                    origin = batch["origin"][i].detach().cpu().view(1, 3)
                    detections[:, :3] += origin
                for z, y, x, radius, score in detections.tolist():
                    self.val_rows.append([seriesuid, z, y, x, radius, score])
                self.log(
                    f"{stage}/detections_per_crop",
                    float(len(detections)),
                    on_step=False,
                    on_epoch=True,
                    batch_size=1,
                    add_dataloader_idx=False,
                )
            return torch.zeros((), device=self.device)
        return self._shared_step(batch, stage)

    def _decode_head(self, outputs: dict[str, torch.Tensor], batch_index: int, head: str = "2") -> torch.Tensor:
        cls = torch.sigmoid(outputs[f"Cls{head}"][batch_index, 0])
        radius = outputs[f"Reg{head}"][batch_index, 0]
        offset = outputs[f"Off{head}"][batch_index]
        mask = cls > self.hparams.decode_threshold
        if not mask.any():
            return cls.new_zeros((0, 5))
        scores = cls[mask]
        if scores.numel() > self.hparams.decode_topk:
            values, order = torch.topk(scores, self.hparams.decode_topk)
            points = mask.nonzero(as_tuple=False)[order]
            scores = values
        else:
            points = mask.nonzero(as_tuple=False)
        off = offset[:, points[:, 0], points[:, 1], points[:, 2]].T
        rad = radius[points[:, 0], points[:, 1], points[:, 2]]
        centers = (points.float() + off) * self.hparams.stride
        return torch.cat([centers, (rad * self.hparams.stride).unsqueeze(1), scores.unsqueeze(1)], dim=1)

    def decode_one(self, outputs: dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        det = torch.cat([self._decode_head(outputs, batch_index, "1"), self._decode_head(outputs, batch_index, "2")], dim=0)
        return sphere_nms(det, self.hparams.nms_threshold, self.hparams.decode_topk)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        outputs = self(batch["image"])
        for i, seriesuid in enumerate(batch["seriesuid"]):
            detections = self.decode_one(outputs, i).detach().cpu()
            if "origin" in batch and len(detections):
                origin = batch["origin"][i].detach().cpu().view(1, 3)
                detections[:, :3] += origin
            for z, y, x, radius, score in detections.tolist():
                self.test_rows.append([seriesuid, z, y, x, radius, score])
            self.log("test/detections_per_crop", float(len(detections)), on_step=False, on_epoch=True, batch_size=1)

    def on_test_epoch_start(self) -> None:
        self.test_rows = []

    def on_validation_epoch_start(self) -> None:
        self.val_rows = []

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        if not self.val_rows:
            return
        out_dir = Path(self.trainer.default_root_dir) / "validation_predictions" / f"epoch_{self.current_epoch + 1:03d}"
        gathered_rows = self._gather_detection_rows(self.val_rows)
        rows = self._merge_detection_rows(gathered_rows, out_dir / "val_predictions.csv")
        evaluate_val_froc = self.hparams.evaluate_froc if self.hparams.evaluate_val_froc is None else self.hparams.evaluate_val_froc
        if evaluate_val_froc:
            self._evaluate_froc(pd.DataFrame(rows, columns=self._prediction_columns()), out_dir, dataset_attr="val_ds", stage="val")

    def on_test_epoch_end(self) -> None:
        out_dir = Path(self.trainer.default_root_dir) / self.hparams.prediction_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        gathered_rows = self._gather_detection_rows(self.test_rows)
        rows = self._merge_detection_rows(gathered_rows, out_dir / "test_predictions.csv")
        if self.trainer.is_global_zero:
            print(f"SCPM-Net test predictions saved to {out_dir / 'test_predictions.csv'}")
        evaluate_test_froc = self.hparams.evaluate_froc if self.hparams.evaluate_test_froc is None else self.hparams.evaluate_test_froc
        if evaluate_test_froc:
            self._evaluate_froc(pd.DataFrame(rows, columns=self._prediction_columns()), out_dir, dataset_attr="test_ds", stage="test")

    @staticmethod
    def _prediction_columns() -> list[str]:
        return ["seriesuid", "coordZ", "coordY", "coordX", "radius", "probability"]

    @staticmethod
    def _gather_detection_rows(rows: list[list[float | str]]) -> list[list[float | str]]:
        if not (dist.is_available() and dist.is_initialized()):
            return rows
        gathered: list[list[list[float | str]]] = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
        dist.all_gather_object(gathered, rows)
        merged: list[list[float | str]] = []
        for rank_rows in gathered:
            merged.extend(rank_rows)
        return merged

    def _merge_detection_rows(self, detection_rows: list[list[float | str]], path: Path) -> list[list[float | str]]:
        if self.trainer.is_global_zero:
            path.parent.mkdir(parents=True, exist_ok=True)
        columns = self._prediction_columns()
        rows = []
        if detection_rows:
            df = pd.DataFrame(detection_rows, columns=columns)
            for seriesuid, group in df.groupby("seriesuid", sort=False):
                detections = torch.as_tensor(
                    group[["coordZ", "coordY", "coordX", "radius", "probability"]].to_numpy(),
                    dtype=torch.float32,
                )
                detections = sphere_nms(detections, self.hparams.nms_threshold, self.hparams.final_topk)
                for z, y, x, radius, score in detections.tolist():
                    rows.append([seriesuid, z, y, x, radius, score])
        if self.trainer.is_global_zero:
            pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
        return rows

    @staticmethod
    def _row_radius(row: pd.Series) -> float:
        if "radius" in row and pd.notna(row["radius"]):
            return float(row["radius"])
        for diameter_col in ("diameter", "diameter_mm"):
            if diameter_col in row and pd.notna(row[diameter_col]):
                return float(row[diameter_col]) / 2.0
        depth_col = "depth" if "depth" in row and pd.notna(row["depth"]) else "d"
        dims = [float(row[col]) for col in ("w", "h", depth_col) if col in row and pd.notna(row[col])]
        return max(dims) / 2.0 if len(dims) == 3 else 0.0

    @classmethod
    def _gt_from_rows(cls, rows: pd.DataFrame) -> np.ndarray:
        if all(col in rows.columns for col in ("coordZ", "coordY", "coordX")):
            coord_cols = ("coordZ", "coordY", "coordX")
        elif all(col in rows.columns for col in ("z", "y", "x")):
            coord_cols = ("z", "y", "x")
        else:
            return np.zeros((0, 4), dtype=np.float32)
        valid = rows.dropna(subset=list(coord_cols))
        if "label" in valid.columns:
            valid = valid[valid["label"].astype(str).str.lower().isin(("nodule", "1", "true", "positive"))]
        spheres = []
        for _, row in valid.iterrows():
            radius = cls._row_radius(row)
            if radius > 0:
                spheres.append([float(row[coord_cols[0]]), float(row[coord_cols[1]]), float(row[coord_cols[2]]), radius])
        return np.asarray(spheres, dtype=np.float32) if spheres else np.zeros((0, 4), dtype=np.float32)

    def _ground_truth_by_series(self, dataset_attr: str = "test_ds") -> tuple[dict[str, np.ndarray], int]:
        datamodule = self.trainer.datamodule if self.trainer else None
        dataset = getattr(datamodule, dataset_attr, None)
        groups = getattr(dataset, "groups", None)
        if not groups:
            return {}, 0
        gt = {}
        for seriesuid, rows in groups:
            gt[str(seriesuid)] = self._gt_from_rows(rows)
        return gt, len(groups)

    @staticmethod
    def _sphere_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(gt[:, :3] - pred[:3], axis=1)
        r1 = max(float(pred[3]), 1e-6)
        r2 = np.maximum(gt[:, 3].astype(np.float64), 1e-6)
        no_overlap = dist >= (r1 + r2)
        contained = (dist + np.minimum(r1, r2)) <= np.maximum(r1, r2)
        inter = np.zeros_like(dist, dtype=np.float64)
        inter[contained] = (4.0 / 3.0) * np.pi * np.minimum(r1, r2[contained]) ** 3
        partial = ~(no_overlap | contained)
        if np.any(partial):
            d = np.maximum(dist[partial], 1e-6)
            rp = r2[partial]
            cos1 = np.clip((r1**2 + d**2 - rp**2) / (2 * r1 * d), -0.999999, 0.999999)
            cos2 = np.clip((rp**2 + d**2 - r1**2) / (2 * rp * d), -0.999999, 0.999999)
            h1 = r1 * (1 - cos1)
            h2 = rp * (1 - cos2)
            inter[partial] = np.pi * h1**2 * (r1 - h1 / 3) + np.pi * h2**2 * (rp - h2 / 3)
        union = (4.0 / 3.0) * np.pi * (r1**3 + r2**3) - inter
        return inter / np.maximum(union, 1e-6)

    def _match_prediction(self, pred: np.ndarray, gt: np.ndarray, matched: np.ndarray) -> int | None:
        available = np.where(~matched)[0]
        if len(available) == 0:
            return None
        available_gt = gt[available]
        strategy = str(self.hparams.froc_match_strategy).lower()
        if strategy == "sphere_iou":
            scores = self._sphere_iou(pred, available_gt)
            best = int(np.argmax(scores))
            return int(available[best]) if scores[best] >= float(self.hparams.froc_iou_threshold) else None
        distances = np.linalg.norm(available_gt[:, :3] - pred[:3], axis=1)
        hits = distances <= available_gt[:, 3]
        if not np.any(hits):
            return None
        hit_indices = np.where(hits)[0]
        best = hit_indices[int(np.argmin(distances[hit_indices]))]
        return int(available[best])

    def _evaluate_froc(self, pred_df: pd.DataFrame, out_dir: Path, dataset_attr: str = "test_ds", stage: str = "test") -> None:
        gt_by_series, num_scans = self._ground_truth_by_series(dataset_attr)
        total_gt = int(sum(len(v) for v in gt_by_series.values()))
        if num_scans == 0 or total_gt == 0:
            print(f"SCPM-Net {stage} FROC skipped: no ground truth available.")
            return

        if pred_df.empty:
            fp_per_scan = np.asarray([0.0])
            sensitivity = np.asarray([0.0])
        else:
            pred_df = pred_df.sort_values("probability", ascending=False).reset_index(drop=True)
            matched = {seriesuid: np.zeros(len(gt), dtype=bool) for seriesuid, gt in gt_by_series.items()}
            tp = 0
            fp = 0
            sensitivities = [0.0]
            fps = [0.0]
            for _, row in pred_df.iterrows():
                seriesuid = str(row["seriesuid"])
                pred = row[["coordZ", "coordY", "coordX", "radius"]].to_numpy(dtype=np.float32)
                gt = gt_by_series.get(seriesuid, np.zeros((0, 4), dtype=np.float32))
                match_idx = self._match_prediction(pred, gt, matched.get(seriesuid, np.zeros(0, dtype=bool)))
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
        froc_values = []
        for rate in tuple(float(v) for v in self.hparams.froc_fp_rates):
            eligible = sensitivity[fp_per_scan <= rate]
            score = float(eligible.max()) if eligible.size else 0.0
            froc_values.append(score)
            rows.append({"fp_per_scan": rate, "sensitivity": score})
            self.log(f"{stage}/froc_{rate:g}fp", score, on_step=False, on_epoch=True, add_dataloader_idx=False)
        mean_froc = float(np.mean(froc_values))
        self.log(f"{stage}/mean_froc", mean_froc, prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        if stage == "val":
            self.log("val_mean_froc", mean_froc, logger=False, on_step=False, on_epoch=True)
            mean_tensor = torch.tensor(mean_froc, device=self.device)
            self.trainer.callback_metrics["val/mean_froc"] = mean_tensor
            self.trainer.callback_metrics["val_mean_froc"] = mean_tensor

        froc_path = out_dir / f"{stage}_froc.csv"
        curve_path = out_dir / f"{stage}_froc_curve.csv"
        if self.trainer.is_global_zero:
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(froc_path, index=False)
            pd.DataFrame({"fp_per_scan": fp_per_scan, "sensitivity": sensitivity}).to_csv(curve_path, index=False)
            print(f"SCPM-Net {stage} FROC saved to {froc_path}; mean={mean_froc:.4f}")

    def configure_optimizers(self):
        optimizer_name = str(self.hparams.optimizer_name).lower()
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer_name: {self.hparams.optimizer_name}")

        warmup_epochs = int(self.hparams.warmup_epochs)
        warmup_lr = float(self.hparams.warmup_lr)
        base_lr = float(self.hparams.lr)
        milestones = tuple(int(v) for v in self.hparams.lr_milestones)
        gamma = float(self.hparams.lr_gamma)

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return warmup_lr / base_lr
            factor = 1.0
            for milestone in milestones:
                if epoch >= milestone:
                    factor *= gamma
            return factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
