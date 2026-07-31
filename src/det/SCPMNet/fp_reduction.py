from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:  # pragma: no cover - lets utility functions import without Lightning installed.
    pl = None

from src.det.SCPMNet.dataset import load_volume, normalize_ct, pad_crop, prepare_label_dataframe
from src.det.SCPMNet.lightning_model import SCPMLitModel


def ground_truth_by_series(csv_path: str | Path, split: str, data_root: str | Path, skip_missing_images: bool = True) -> dict[str, np.ndarray]:
    groups = prepare_label_dataframe(str(csv_path), split, Path(data_root), skip_missing_images)
    return {str(seriesuid): SCPMLitModel._gt_from_rows(rows) for seriesuid, rows in groups.groupby("seriesuid", sort=False)}


def image_paths_by_series(csv_path: str | Path, split: str, data_root: str | Path, skip_missing_images: bool = True) -> dict[str, Path]:
    df = prepare_label_dataframe(str(csv_path), split, Path(data_root), skip_missing_images)
    return {str(seriesuid): Path(rows.iloc[0]["_resolved_image_path"]) for seriesuid, rows in df.groupby("seriesuid", sort=False)}


def label_candidates(
    candidates: pd.DataFrame,
    gt_by_series: dict[str, np.ndarray],
    ignore_margin: float = 2.0,
) -> pd.DataFrame:
    rows = []
    for _, row in candidates.iterrows():
        seriesuid = str(row["seriesuid"])
        gt = gt_by_series.get(seriesuid, np.zeros((0, 4), dtype=np.float32))
        label = 0
        nearest_gt_index = -1
        nearest_distance = np.nan
        nearest_gt_radius = np.nan
        ignored = False
        if len(gt):
            pred_center = row[["coordZ", "coordY", "coordX"]].to_numpy(dtype=np.float32)
            distances = np.linalg.norm(gt[:, :3] - pred_center.reshape(1, 3), axis=1)
            nearest_gt_index = int(np.argmin(distances))
            nearest_distance = float(distances[nearest_gt_index])
            nearest_gt_radius = float(gt[nearest_gt_index, 3])
            if nearest_distance <= nearest_gt_radius:
                label = 1
            elif nearest_distance <= nearest_gt_radius + ignore_margin:
                ignored = True
        item = row.to_dict()
        item.update(
            {
                "label": label,
                "ignore": ignored,
                "nearest_gt_index": nearest_gt_index,
                "nearest_gt_distance": nearest_distance,
                "nearest_gt_radius": nearest_gt_radius,
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def evaluate_froc(
    pred_df: pd.DataFrame,
    gt_by_series: dict[str, np.ndarray],
    fp_rates: Sequence[float] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    score_col: str = "probability",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    total_gt = int(sum(len(v) for v in gt_by_series.values()))
    num_scans = int(len(gt_by_series))
    if total_gt == 0 or num_scans == 0:
        raise ValueError("Cannot evaluate FROC without ground truth scans and nodules.")

    if pred_df.empty:
        fp_per_scan = np.asarray([0.0], dtype=np.float32)
        sensitivity = np.asarray([0.0], dtype=np.float32)
    else:
        pred_df = pred_df.sort_values(score_col, ascending=False).reset_index(drop=True)
        matched = {seriesuid: np.zeros(len(gt), dtype=bool) for seriesuid, gt in gt_by_series.items()}
        tp = 0
        fp = 0
        sensitivities = [0.0]
        fps = [0.0]
        for _, row in pred_df.iterrows():
            seriesuid = str(row["seriesuid"])
            gt = gt_by_series.get(seriesuid, np.zeros((0, 4), dtype=np.float32))
            pred_center = row[["coordZ", "coordY", "coordX"]].to_numpy(dtype=np.float32)
            available = np.where(~matched.get(seriesuid, np.zeros(0, dtype=bool)))[0]
            match_idx = None
            if len(available):
                available_gt = gt[available]
                distances = np.linalg.norm(available_gt[:, :3] - pred_center.reshape(1, 3), axis=1)
                hits = distances <= available_gt[:, 3]
                if np.any(hits):
                    hit_indices = np.where(hits)[0]
                    best = hit_indices[int(np.argmin(distances[hit_indices]))]
                    match_idx = int(available[best])
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
    for rate in fp_rates:
        eligible = sensitivity[fp_per_scan <= float(rate)]
        value = float(eligible.max()) if eligible.size else 0.0
        values.append(value)
        rows.append({"fp_per_scan": float(rate), "sensitivity": value})
    curve = pd.DataFrame({"fp_per_scan": fp_per_scan, "sensitivity": sensitivity})
    summary = pd.DataFrame(rows)
    return summary, curve, float(np.mean(values))


class CandidatePatchDataset(Dataset):
    def __init__(
        self,
        candidates_csv: str | Path,
        csv_path: str | Path,
        split: str,
        data_root: str | Path,
        patch_size: Sequence[int] = (32, 32, 32),
        clip: Sequence[float] = (-1000.0, 400.0),
        intensity_mode: str = "hu",
        skip_missing_images: bool = True,
        include_ignored: bool = False,
        augment: bool = False,
        volume_cache_size: int = 4,
        normalized_volume_cache_dir: str | Path | None = None,
    ):
        self.candidates = pd.read_csv(candidates_csv)
        if "label" not in self.candidates.columns:
            raise ValueError(f"{candidates_csv} must contain a label column. Generate candidates with --label-candidates.")
        if "ignore" in self.candidates.columns and not include_ignored:
            self.candidates = self.candidates[~self.candidates["ignore"].astype(bool)].copy()
        self.candidates = self.candidates.reset_index(drop=True)
        self.image_paths = image_paths_by_series(csv_path, split, data_root, skip_missing_images)
        self.patch_size = np.asarray(tuple(int(v) for v in patch_size), dtype=np.int64)
        self.clip = (float(clip[0]), float(clip[1]))
        self.intensity_mode = str(intensity_mode)
        self.augment = augment
        self.volume_cache_size = max(1, int(volume_cache_size))
        self.normalized_volume_cache_dir = Path(normalized_volume_cache_dir) if normalized_volume_cache_dir else None
        if self.normalized_volume_cache_dir is not None:
            self.normalized_volume_cache_dir.mkdir(parents=True, exist_ok=True)
        self._volume_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.candidates)

    def labels(self) -> np.ndarray:
        return self.candidates["label"].to_numpy(dtype=np.int64)

    def _load_volume(self, seriesuid: str) -> np.ndarray:
        if seriesuid in self._volume_cache:
            volume = self._volume_cache.pop(seriesuid)
            self._volume_cache[seriesuid] = volume
            return volume
        volume = self._load_normalized_volume(seriesuid)
        self._volume_cache[seriesuid] = volume
        while len(self._volume_cache) > self.volume_cache_size:
            self._volume_cache.popitem(last=False)
        return volume

    def _normalized_cache_path(self, seriesuid: str) -> Path:
        if self.normalized_volume_cache_dir is None:
            raise RuntimeError("normalized_volume_cache_dir is not configured.")
        safe_seriesuid = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in seriesuid)
        clip_tag = f"{self.clip[0]:g}_{self.clip[1]:g}".replace("-", "m").replace(".", "p")
        return self.normalized_volume_cache_dir / f"{safe_seriesuid}_mode-{self.intensity_mode}_clip-{clip_tag}.npy"

    def _load_normalized_volume(self, seriesuid: str) -> np.ndarray:
        if self.normalized_volume_cache_dir is None:
            return normalize_ct(load_volume(self.image_paths[seriesuid]), self.clip, self.intensity_mode)

        cache_path = self._normalized_cache_path(seriesuid)
        if cache_path.exists():
            return np.load(cache_path, mmap_mode="r")

        volume = normalize_ct(load_volume(self.image_paths[seriesuid]), self.clip, self.intensity_mode)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}.npy")
        try:
            np.save(tmp_path, volume)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return np.load(cache_path, mmap_mode="r")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.candidates.iloc[index]
        seriesuid = str(row["seriesuid"])
        volume = self._load_volume(seriesuid)
        center = row[["coordZ", "coordY", "coordX"]].to_numpy(dtype=np.float32)
        start = np.floor(center - self.patch_size / 2).astype(np.int64)
        patch, _ = pad_crop(volume, start, self.patch_size)
        if self.augment:
            for axis in range(3):
                if np.random.rand() < 0.5:
                    patch = np.flip(patch, axis=axis).copy()
            if np.random.rand() < 0.25:
                patch = patch + np.random.normal(0.0, 0.03, size=patch.shape).astype(np.float32)
                patch = np.clip(patch, -1.0, 1.0)
        meta = np.asarray(
            [
                float(row.get("probability", 0.0)),
                float(row.get("radius", 0.0)) / 32.0,
                float(center[0]) / max(volume.shape[0], 1),
                float(center[1]) / max(volume.shape[1], 1),
                float(center[2]) / max(volume.shape[2], 1),
            ],
            dtype=np.float32,
        )
        return {
            "image": torch.from_numpy(patch[None].astype(np.float32)),
            "meta": torch.from_numpy(meta),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }


class FPReductionNet(nn.Module):
    def __init__(self, meta_dim: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 + meta_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        feat = self.features(image).flatten(1)
        return self.classifier(torch.cat([feat, meta], dim=1)).squeeze(1)


class FPReductionLitModel(pl.LightningModule if pl is not None else nn.Module):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: float | None = None,
        loss_name: str = "bce",
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        if pl is not None:
            self.save_hyperparameters()
        self.net = FPReductionNet()
        self.lr = lr
        self.weight_decay = weight_decay
        self.pos_weight_value = pos_weight
        self.loss_name = str(loss_name).lower()
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self._epoch_probs: dict[str, list[torch.Tensor]] = {"train": [], "val": []}
        self._epoch_labels: dict[str, list[torch.Tensor]] = {"train": [], "val": []}

    def forward(self, image: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return self.net(image, meta)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_weight = None
        if self.pos_weight_value is not None:
            pos_weight = torch.tensor(float(self.pos_weight_value), device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
        if self.loss_name == "bce":
            return bce.mean()
        if self.loss_name != "focal":
            raise ValueError(f"Unsupported FPR loss: {self.loss_name}")

        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
        alpha_t = self.focal_alpha * labels + (1.0 - self.focal_alpha) * (1.0 - labels)
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.focal_gamma)
        return (focal_weight * bce).mean()

    @staticmethod
    def _binary_mcc(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        preds = probs >= 0.5
        labels_bool = labels >= 0.5
        tp = (preds & labels_bool).sum().float()
        tn = (~preds & ~labels_bool).sum().float()
        fp = (preds & ~labels_bool).sum().float()
        fn = (~preds & labels_bool).sum().float()
        denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom <= 0:
            return torch.zeros((), dtype=torch.float32, device=probs.device)
        return (tp * tn - fp * fn) / denom

    @staticmethod
    def _binary_auc(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = (labels >= 0.5).long()
        n_pos = labels.sum().float()
        n_neg = (labels.numel() - labels.sum()).float()
        if n_pos <= 0 or n_neg <= 0:
            return torch.full((), float("nan"), dtype=torch.float32, device=probs.device)

        sorted_probs, order = torch.sort(probs)
        sorted_labels = labels[order]
        ranks = torch.arange(1, probs.numel() + 1, dtype=torch.float32, device=probs.device)
        _, inverse, counts = torch.unique_consecutive(sorted_probs, return_inverse=True, return_counts=True)
        if torch.any(counts > 1):
            starts = torch.cumsum(counts, dim=0) - counts
            avg_ranks = starts.float() + (counts.float() + 1.0) / 2.0
            ranks = avg_ranks[inverse]

        pos_rank_sum = ranks[sorted_labels == 1].sum()
        return (pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)

    def _reset_epoch_metrics(self, stage: str) -> None:
        self._epoch_probs[stage] = []
        self._epoch_labels[stage] = []

    def _log_epoch_metrics(self, stage: str) -> None:
        if not self._epoch_probs[stage]:
            return
        probs = torch.cat(self._epoch_probs[stage]).float()
        labels = torch.cat(self._epoch_labels[stage]).float()
        mcc = self._binary_mcc(probs, labels)
        auc = self._binary_auc(probs, labels)
        if pl is not None:
            self.log(f"{stage}/mcc", mcc, prog_bar=stage == "val", on_epoch=True)
            self.log(f"{stage}/auc", auc, prog_bar=stage == "val", on_epoch=True)
        self._reset_epoch_metrics(stage)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits = self(batch["image"], batch["meta"])
        labels = batch["label"]
        loss = self._loss(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        acc = (preds == labels).float().mean()
        self._epoch_probs[stage].append(probs.detach().cpu())
        self._epoch_labels[stage].append(labels.detach().cpu())
        if pl is not None:
            self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, batch_size=labels.numel())
            self.log(f"{stage}/acc", acc, prog_bar=stage == "val", on_epoch=True, batch_size=labels.numel())
        return loss

    def on_train_epoch_start(self) -> None:
        self._reset_epoch_metrics("train")

    def on_validation_epoch_start(self) -> None:
        self._reset_epoch_metrics("val")

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
