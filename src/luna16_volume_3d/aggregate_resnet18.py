from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.video import r3d_18


class Luna16VolumeDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split_csv: str | Path,
        split: str,
        volume_size: tuple[int, int, int],
        train: bool,
    ) -> None:
        self.data_root = Path(data_root)
        self.split_csv = Path(split_csv)
        self.volume_size = volume_size
        self.train = train

        df = pd.read_csv(self.split_csv)
        df = df[(df["split"].astype(str) == split) & (df["target"].isin([0, 1]))].copy()
        df["target"] = df["target"].astype(int)
        if df.empty:
            raise RuntimeError(f"No binary samples found for split={split} in {self.split_csv}")
        self.df = df.reset_index(drop=True)
        self.labels = self.df["target"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        path = self.data_root / row["image_path"]
        image = sitk.ReadImage(str(path))
        volume = sitk.GetArrayFromImage(image).astype(np.float32)
        volume = np.nan_to_num(volume, nan=0.0, posinf=255.0, neginf=0.0)
        volume = np.clip(volume, 0.0, 255.0) / 255.0

        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=self.volume_size, mode="trilinear", align_corners=False)
        tensor = tensor.squeeze(0)

        if self.train:
            if torch.rand(()) < 0.5:
                tensor = torch.flip(tensor, dims=[3])
            if torch.rand(()) < 0.5:
                tensor = torch.flip(tensor, dims=[2])
            if torch.rand(()) < 0.25:
                noise = torch.randn_like(tensor) * 0.015
                tensor = (tensor + noise).clamp(0.0, 1.0)

        label = torch.tensor(int(row["target"]), dtype=torch.long)
        return tensor, label, str(row["seriesuid"])

    def sampler(self) -> WeightedRandomSampler:
        labels = torch.tensor(self.labels, dtype=torch.long)
        counts = torch.bincount(labels, minlength=2).float().clamp_min(1.0)
        weights = 1.0 / counts[labels]
        return WeightedRandomSampler(weights.double(), num_samples=len(weights), replacement=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_model() -> nn.Module:
    model = r3d_18(weights=None)
    model.stem[0] = nn.Conv3d(
        1,
        64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(1, 3, 3),
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype | None):
    model.eval()
    losses: list[float] = []
    labels: list[int] = []
    preds: list[int] = []
    scores: list[float] = []
    sample_ids: list[str] = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for volumes, y, ids in loader:
            volumes = volumes.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(volumes)
                loss = criterion(logits, y)
            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            pred = torch.argmax(logits.float(), dim=1)
            losses.append(float(loss.detach().cpu()) * y.numel())
            labels.extend(y.detach().cpu().tolist())
            preds.extend(pred.detach().cpu().tolist())
            scores.extend(prob.detach().cpu().tolist())
            sample_ids.extend(ids)

    return compute_metrics(labels, preds, scores, sum(losses) / max(1, len(labels))), sample_ids, labels, preds, scores


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
    parser = argparse.ArgumentParser(description="Train a full-volume 3D ResNet18 LUNA16 malignancy baseline.")
    parser.add_argument("--data-root", default="/ssd2/domenico/datasets/LUNA16_preprocessed")
    parser.add_argument("--splits-dir", default="/ssd2/domenico/datasets/LUNA16_preprocessed/cv_splits")
    parser.add_argument("--output-dir", default="outputs/luna16_volume_3d_resnet18")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--volume-size", type=int, nargs=3, default=[96, 160, 160], metavar=("D", "H", "W"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["32", "16-mixed", "bf16-mixed"], default="16-mixed")
    parser.add_argument("--monitor", choices=["val_auc", "val_mcc", "val_loss"], default="val_mcc")
    parser.add_argument("--patience", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed + args.fold)

    output_dir = Path(args.output_dir) / f"fold_{args.fold}" / "resnet18_3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    split_csv = Path(args.splits_dir) / f"luna16_classification_fold{args.fold}.csv"
    train_ds = Luna16VolumeDataset(args.data_root, split_csv, "train", tuple(args.volume_size), train=True)
    val_ds = Luna16VolumeDataset(args.data_root, split_csv, "val", tuple(args.volume_size), train=False)
    test_ds = Luna16VolumeDataset(args.data_root, split_csv, "test", tuple(args.volume_size), train=False)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, sampler=train_ds.sampler(), drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp_dtype = None
    if device.type == "cuda" and args.precision == "16-mixed":
        amp_dtype = torch.float16
    elif device.type == "cuda" and args.precision == "bf16-mixed":
        amp_dtype = torch.bfloat16

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    criterion = nn.CrossEntropyLoss()

    best_score = -math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / "best.pt"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for volumes, y, _ in train_loader:
            volumes = volumes.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(volumes)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu()) * y.numel()
            seen += y.numel()
        scheduler.step()

        val_metrics, _, _, _, _ = evaluate(model, val_loader, device, amp_dtype)
        train_loss = running_loss / max(1, seen)
        score = metric_for_monitor(val_metrics, args.monitor)
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "monitor": args.monitor, "score": score}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}", flush=True)
                break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, sample_ids, labels, preds, scores = evaluate(model, test_loader, device, amp_dtype)
    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    test_metrics["best_epoch"] = int(checkpoint["epoch"])
    test_metrics["best_monitor_score"] = float(checkpoint["score"])
    with (output_dir / "test_metrics.json").open("w") as handle:
        json.dump(test_metrics, handle, indent=2)

    with (output_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fold", "backbone", "sample_id", "split", "label", "prediction", "score"],
        )
        writer.writeheader()
        for sid, label, pred, score in zip(sample_ids, labels, preds, scores):
            writer.writerow(
                {
                    "fold": args.fold,
                    "backbone": "resnet18_3d",
                    "sample_id": sid,
                    "split": "test",
                    "label": label,
                    "prediction": pred,
                    "score": score,
                }
            )


if __name__ == "__main__":
    main()
