"""Train/evaluate a frozen-RadJEPA-3D linear probe on one LUNA16 CV fold.

Loads the cached per-window RadJEPA-3D embeddings produced by
``extract_features.py`` and derives one of four patient-level representations
(no re-running the encoder):

  - ``resize32``: the single whole-volume-resized-to-32-slices embedding.
  - ``sliding_mean``: element-wise mean of the sliding-window embeddings.
  - ``sliding_max``: element-wise max of the sliding-window embeddings.
  - ``sliding_mean_max``: concatenation of the mean and max above.

A single ``Linear(feature_dim, 1)`` head (BCEWithLogitsLoss) is trained on
top, using the *exact* same fold CSV (``luna16_classification_fold{fold}.csv``)
already used by the Adaptive RBF experiments. It reuses representation-agnostic
training and evaluation utilities shared by the volumetric baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.luna16_foundation_common import (  # noqa: E402
    build_model,
    compute_metrics,
    evaluate,
    metric_for_monitor,
    seed_everything,
)

REPRESENTATIONS = ["resize32", "sliding_mean", "sliding_max", "sliding_mean_max"]


def patient_representation(cache: dict, representation: str) -> np.ndarray:
    if representation == "resize32":
        return cache["resize32_embedding"].astype(np.float32)
    windows = cache["window_embeddings"].astype(np.float32)
    if representation == "sliding_mean":
        return windows.mean(axis=0)
    if representation == "sliding_max":
        return windows.max(axis=0)
    if representation == "sliding_mean_max":
        return np.concatenate([windows.mean(axis=0), windows.max(axis=0)], axis=0)
    raise ValueError(f"Unknown representation: {representation}")


class RadJepaEmbeddingDataset(Dataset):
    def __init__(self, per_patient_dir: Path, split_csv: Path, split: str, representation: str) -> None:
        df = pd.read_csv(split_csv)
        df = df[(df["split"].astype(str) == split) & (df["target"].isin([0, 1]))].copy()
        df["target"] = df["target"].astype(int)
        if df.empty:
            raise RuntimeError(f"No binary samples found for split={split} in {split_csv}")
        df = df.reset_index(drop=True)

        features = []
        missing = []
        for seriesuid in df["seriesuid"]:
            cache_path = per_patient_dir / f"{seriesuid}.npz"
            if not cache_path.exists():
                missing.append(seriesuid)
                continue
            cache = np.load(cache_path)
            features.append(patient_representation(cache, representation))
        if missing:
            raise RuntimeError(
                f"{len(missing)} seriesuid(s) in {split_csv} (split={split}) have no cached "
                f"RadJEPA-3D embedding, e.g. {missing[:5]}. Run extract_features.py first."
            )

        self.df = df
        self.features = np.stack(features).astype(np.float32)
        self.labels = self.df["target"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        feature = torch.from_numpy(self.features[index])
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return feature, label, str(self.df.iloc[index]["seriesuid"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-RadJEPA-3D linear probe on one LUNA16 fold.")
    parser.add_argument("--embeddings-dir", default="outputs/luna16_radjepa_3d/embeddings/per_patient")
    parser.add_argument("--splits-dir", default="data/processed/cv_splits")
    parser.add_argument("--output-dir", default="outputs/luna16_radjepa_3d/probes")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--monitor", choices=["val_auc", "val_mcc", "val_loss"], default="val_mcc")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Run all epochs and only use the monitor for best-checkpoint selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed + args.fold)

    output_dir = Path(args.output_dir) / f"fold_{args.fold}" / args.representation
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    split_csv = Path(args.splits_dir) / f"luna16_classification_fold{args.fold}.csv"
    per_patient_dir = Path(args.embeddings_dir)

    train_ds = RadJepaEmbeddingDataset(per_patient_dir, split_csv, "train", args.representation)
    val_ds = RadJepaEmbeddingDataset(per_patient_dir, split_csv, "val", args.representation)
    test_ds = RadJepaEmbeddingDataset(per_patient_dir, split_csv, "test", args.representation)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    in_features = train_ds.features.shape[1]
    model = build_model(in_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    labels_arr = np.array(train_ds.labels)
    n_pos = max(1, int(labels_arr.sum()))
    n_neg = max(1, int(len(labels_arr) - labels_arr.sum()))
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)
    train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_criterion = nn.BCEWithLogitsLoss()

    best_score = -math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / "best.pt"

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        seen = 0
        for features, y, _ in train_loader:
            features = features.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(features).squeeze(1)
            loss = train_criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * y.numel()
            seen += y.numel()
        scheduler.step()

        val_metrics, _, _, _, _ = evaluate(model, val_loader, device, eval_criterion)
        train_loss = running_loss / max(1, seen)
        score = metric_for_monitor(val_metrics, args.monitor)
        row = {
            "epoch": epoch,
            "epoch_seconds": time.time() - epoch_start,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "monitor": args.monitor, "score": score},
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if not args.no_early_stopping and stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}", flush=True)
                break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, sample_ids, labels, preds, scores = evaluate(model, test_loader, device, eval_criterion)
    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    test_metrics["best_epoch"] = int(checkpoint["epoch"])
    test_metrics["best_monitor_score"] = float(checkpoint["score"])
    test_metrics["feature_dim"] = int(in_features)
    with (output_dir / "test_metrics.json").open("w") as handle:
        json.dump(test_metrics, handle, indent=2)

    with (output_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fold", "representation", "sample_id", "split", "label", "prediction", "score"],
        )
        writer.writeheader()
        for sid, label, pred, score in zip(sample_ids, labels, preds, scores):
            writer.writerow(
                {
                    "fold": args.fold,
                    "representation": args.representation,
                    "sample_id": sid,
                    "split": "test",
                    "label": label,
                    "prediction": pred,
                    "score": score,
                }
            )

    print(json.dumps({"fold": args.fold, "representation": args.representation, **test_metrics}), flush=True)


if __name__ == "__main__":
    main()
