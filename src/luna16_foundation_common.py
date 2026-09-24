"""Shared data, metric, and linear-probe helpers for volumetric baselines."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


PER_FOLD_METRIC_KEYS = ["auc", "mcc", "acc", "f1", "precision", "recall"]


def find_raw_ct_path(raw_root: Path, image_path: str) -> Path:
    """Map a processed-volume path to its raw LUNA16 ``.mhd`` file."""
    subset = image_path.split("/")[0]
    seriesuid = Path(image_path).name.removesuffix("_volume.nii.gz")
    return raw_root / subset / subset / f"{seriesuid}.mhd"


def load_patient_table(fold_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(fold_csv)
    required = {"seriesuid", "image_path", "target", "target_name"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{fold_csv} is missing expected columns: {missing}")
    return df.drop_duplicates(subset="seriesuid").reset_index(drop=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(in_features: int) -> nn.Module:
    return nn.Linear(in_features, 1)


def compute_metrics(
    labels: list[int], preds: list[int], scores: list[float], loss: float
) -> dict[str, float]:
    auc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan")
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "loss": float(loss),
        "auc": float(auc),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    losses: list[float] = []
    labels: list[int] = []
    preds: list[int] = []
    scores: list[float] = []
    sample_ids: list[str] = []
    for features, y, ids in loader:
        features = features.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(features).squeeze(1)
        loss = criterion(logits, y)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long()
        losses.append(float(loss.detach().cpu()) * y.numel())
        labels.extend(y.long().detach().cpu().tolist())
        preds.extend(pred.detach().cpu().tolist())
        scores.extend(prob.detach().cpu().tolist())
        sample_ids.extend(ids)
    metrics = compute_metrics(labels, preds, scores, sum(losses) / max(1, len(labels)))
    return metrics, sample_ids, labels, preds, scores


def metric_for_monitor(metrics: dict[str, float], monitor: str) -> float:
    if monitor == "val_loss":
        return -metrics["loss"]
    return metrics[monitor.replace("val_", "")]


def pooled_metric_row(predictions: pd.DataFrame, representation: str) -> dict[str, object]:
    y = predictions["label"].astype(int).to_numpy()
    pred = predictions["prediction"].astype(int).to_numpy()
    score = predictions["score"].astype(float).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "representation": representation,
        "samples": int(len(predictions)),
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
        "auc": float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def per_fold_summary(output_dir: Path, representation: str) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(output_dir.glob(f"fold_*/{representation}/test_metrics.json")):
        fold = int(metrics_path.parent.parent.name.split("_")[1])
        with metrics_path.open() as handle:
            metrics = json.load(handle)
        row = {"fold": fold}
        for key in PER_FOLD_METRIC_KEYS:
            row[key] = metrics[f"test_{key}"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
