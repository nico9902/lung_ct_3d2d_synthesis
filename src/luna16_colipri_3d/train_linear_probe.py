"""Train/evaluate a frozen-COLIPRI linear probe on one LUNA16 CV fold.

Loads the cached 768-D COLIPRI embeddings produced by ``extract_features.py``
and trains a single ``Linear(768, 1)`` head (BCEWithLogitsLoss) on top of one
of the two frozen representations ("pooled" or "dense_maxpool"), using the
*exact* same fold CSV (``luna16_classification_fold{fold}.csv``) already used
by the Adaptive RBF experiments for the train/val/test split. No new folds are
created, so there is no patient leakage beyond what the original split already
guarantees (LUNA16 subset-level grouping).

Mirrors the metrics/early-stopping/output-file conventions of
``src/luna16_volume_3d/train_resnet18.py`` for drop-in compatibility with the
same downstream aggregation style.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
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
from torch.utils.data import DataLoader, Dataset


class ColipriEmbeddingDataset(Dataset):
    def __init__(self, embeddings_npz: Path, split_csv: Path, split: str, representation: str) -> None:
        cache = np.load(embeddings_npz, allow_pickle=True)
        emb_by_uid = {
            uid: cache[representation][i] for i, uid in enumerate(cache["seriesuid"])
        }

        df = pd.read_csv(split_csv)
        df = df[(df["split"].astype(str) == split) & (df["target"].isin([0, 1]))].copy()
        df["target"] = df["target"].astype(int)
        missing = [uid for uid in df["seriesuid"] if uid not in emb_by_uid]
        if missing:
            raise RuntimeError(
                f"{len(missing)} seriesuid(s) in {split_csv} (split={split}) have no cached "
                f"COLIPRI embedding, e.g. {missing[:5]}. Run extract_features.py first."
            )
        if df.empty:
            raise RuntimeError(f"No binary samples found for split={split} in {split_csv}")

        self.df = df.reset_index(drop=True)
        self.features = np.stack([emb_by_uid[uid] for uid in self.df["seriesuid"]]).astype(np.float32)
        self.labels = self.df["target"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        feature = torch.from_numpy(self.features[index])
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return feature, label, str(self.df.iloc[index]["seriesuid"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(in_features: int) -> nn.Module:
    return nn.Linear(in_features, 1)


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


def compute_metrics(labels: list[int], preds: list[int], scores: list[float], loss: float) -> dict[str, float]:
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


def metric_for_monitor(metrics: dict[str, float], monitor: str) -> float:
    if monitor == "val_loss":
        return -metrics["loss"]
    return metrics[monitor.replace("val_", "")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-COLIPRI linear probe on one LUNA16 fold.")
    parser.add_argument("--embeddings", default="outputs/luna16_colipri_3d/embeddings/colipri_embeddings.npz")
    parser.add_argument("--splits-dir", default="data/processed/cv_splits")
    parser.add_argument("--output-dir", default="outputs/luna16_colipri_3d/probes")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--representation", choices=["pooled", "dense_maxpool"], required=True)
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
        json.dump(vars(args), handle, indent=2)

    split_csv = Path(args.splits_dir) / f"luna16_classification_fold{args.fold}.csv"
    embeddings_npz = Path(args.embeddings)

    train_ds = ColipriEmbeddingDataset(embeddings_npz, split_csv, "train", args.representation)
    val_ds = ColipriEmbeddingDataset(embeddings_npz, split_csv, "val", args.representation)
    test_ds = ColipriEmbeddingDataset(embeddings_npz, split_csv, "test", args.representation)

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
