"""Frozen 3DINO linear probes on full-volume and detector-centred 112^3 crops.

Run ``--variant both`` to extract fold-independent features and fit all ten
classification folds. Checkpoints are selected by validation MCC. Detector
coordinates are voxel x/y/z on the 1 mm processed CT grid, as in the
synthetic-image pipeline; no reference-nodule coordinates are used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.luna16_3dino_3d.model import (  # noqa: E402
    EMBED_DIM,
    DEFAULT_3DINO_CODE_DIR,
    extract_windows,
    NATIVE_INPUT_SIZE,
    build_backbone,
    load_checkpoint,
    percentile_normalize,
)


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code-dir", type=Path, default=DEFAULT_3DINO_CODE_DIR)
    p.add_argument("--variant", choices=["whole", "detections", "both"], default="both")
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--splits-dir", type=Path, default=Path("data/processed/cv_splits"))
    p.add_argument("--detector-csv", type=Path, default=Path("outputs/cpmnetv2_paper_detector_all_oof_predictions.csv"))
    p.add_argument("--checkpoint-path", type=Path, default=Path("external/3DINO-ViT/3dino_vit_weights.pth"))
    p.add_argument("--windows-cache-dir", type=Path, default=Path("outputs/luna16_3dino_3d/window_cache"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/luna16_3dino_frozen_probe"))
    p.add_argument("--folds", type=int, nargs="*", default=list(range(10)))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=233)
    p.add_argument("--micro-batch-windows", type=int, default=32)
    p.add_argument("--skip-extraction", action="store_true")
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--num-shards", type=int, default=1, help="Parallel feature-extraction shards.")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return p.parse_args()


def classification_rows(splits_dir: Path) -> pd.DataFrame:
    rows = pd.read_csv(splits_dir / "luna16_classification_fold0.csv")
    rows = rows[rows.target.isin([0, 1])].copy()
    if rows.seriesuid.duplicated().any():
        raise ValueError("Duplicate classification seriesuid")
    return rows


def detector_candidates(path: Path, uids: set[str]) -> dict[str, pd.DataFrame]:
    rows = pd.read_csv(path, dtype={"seriesuid": str})
    required = {"seriesuid", "fold", "x", "y", "z", "prob_nodule"}
    if not required.issubset(rows):
        raise ValueError(f"Detector CSV missing {required - set(rows)}")
    if not uids.issubset(set(rows.seriesuid)):
        raise ValueError(f"Missing detector predictions for {len(uids - set(rows.seriesuid))} scans")
    rows = rows[rows.seriesuid.isin(uids)].copy()
    if (rows.groupby("seriesuid").fold.nunique() != 1).any():
        raise ValueError("Detector predictions are not unique to an out-of-fold detector")
    out = {}
    for uid, group in rows.groupby("seriesuid"):
        group = group.sort_values("prob_nodule", ascending=False)
        selected = group[group.prob_nodule >= 0.50].head(4)
        # A score-only fallback gives every scan an image-derived crop and
        # never branches on its classification label.
        out[uid] = selected if len(selected) else group.head(1)
    return out


def validate_detector_oof_folds(candidates: dict[str, pd.DataFrame], splits_dir: Path) -> None:
    for fold in range(10):
        rows = pd.read_csv(splits_dir / f"luna16_classification_fold{fold}.csv", dtype={"seriesuid": str})
        for uid in rows.loc[(rows.split == "test") & rows.target.isin([0, 1]), "seriesuid"]:
            if int(candidates[uid].iloc[0].fold) != fold:
                raise ValueError(f"Detector fold for {uid} does not match classification test fold {fold}")


def centered_crop(volume: torch.Tensor, xyz: tuple[float, float, float], size: int = NATIVE_INPUT_SIZE) -> torch.Tensor:
    """Extract z/y/x crop; pad outside a volume with zero (air in this CT encoding)."""
    starts = [int(round(c)) - size // 2 for c in xyz[::-1]]
    result = volume.new_zeros((size, size, size))
    source = []
    target = []
    for start, length in zip(starts, volume.shape):
        lo, hi = max(start, 0), min(start + size, length)
        if hi <= lo:
            raise ValueError(f"Detection centre {xyz} does not intersect volume {tuple(volume.shape)}")
        source.append(slice(lo, hi))
        target.append(slice(lo - start, hi - start))
    result[tuple(target)] = volume[tuple(source)]
    return result


def encode_chunks(backbone: nn.Module, windows: torch.Tensor, device: str, chunk_size: int) -> torch.Tensor:
    embeddings = []
    with torch.inference_mode():
        for chunk in windows.split(chunk_size):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                embeddings.append(backbone(chunk.to(device)).float().cpu())
    features = torch.cat(embeddings)
    return torch.cat((features.mean(0), features.max(0).values)).contiguous()


def extract_features(args: argparse.Namespace, variant: str, rows: pd.DataFrame, candidates: dict[str, pd.DataFrame] | None) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: frozen 3DINO feature extraction requires a visible NVIDIA GPU")
    feature_dir = args.output_dir / variant / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    shard_rows = rows.iloc[args.shard_index::args.num_shards]
    pending = shard_rows[~shard_rows.seriesuid.map(lambda uid: (feature_dir / f"{uid}.pt").exists())]
    if pending.empty:
        print(f"[{variant}] all {len(rows)} scan features already cached", flush=True)
        return
    backbone = build_backbone(args.code_dir)
    load_checkpoint(backbone, args.checkpoint_path, args.code_dir)
    backbone.requires_grad_(False).eval().to(args.device)
    for index, (_, row) in enumerate(pending.iterrows(), 1):
        uid = str(row.seriesuid)
        if variant == "whole":
            path = args.windows_cache_dir / f"{uid}_w112_s56.pt"
            if path.exists():
                windows = torch.load(path, map_location="cpu", weights_only=True).float()
            else:
                import SimpleITK as sitk

                volume = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(str(args.data_root / row.image_path))).astype(np.float32))
                volume = volume.nan_to_num(nan=0.0, posinf=255.0, neginf=0.0).clamp_(0, 255).div_(255)
                windows, _ = extract_windows(volume, NATIVE_INPUT_SIZE, NATIVE_INPUT_SIZE // 2)
                windows = percentile_normalize(windows)
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(f".{os.getpid()}.tmp")
                torch.save(windows.half(), temp)
                temp.replace(path)
                windows = windows.half().float()
            n_windows = len(windows)
            fallback = False
        else:
            # Direct volume loading is needed because fold0 splits scans among train/val/test.
            import SimpleITK as sitk

            volume = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(str(args.data_root / row.image_path))).astype(np.float32))
            volume = volume.nan_to_num(nan=0.0, posinf=255.0, neginf=0.0).clamp_(0, 255).div_(255)
            selected = candidates[uid]
            windows = torch.stack([
                centered_crop(volume, (float(r.x), float(r.y), float(r.z)))
                for r in selected.itertuples()
            ]).unsqueeze(1)
            windows = percentile_normalize(windows)
            n_windows = len(windows)
            fallback = bool(float(selected.iloc[0].prob_nodule) < 0.50)
        feature = encode_chunks(backbone, windows, args.device, args.micro_batch_windows)
        if feature.shape != (2 * EMBED_DIM,) or not torch.isfinite(feature).all():
            raise ValueError(f"Invalid 3DINO feature for {uid}: {tuple(feature.shape)}")
        record = {"seriesuid": uid, "feature": feature, "n_windows": n_windows, "fallback": fallback}
        path = feature_dir / f"{uid}.pt"
        temp = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(record, temp)
        temp.replace(path)
        print(f"[{variant}] {index}/{len(pending)} {uid}: {n_windows} crops", flush=True)


def load_features(feature_dir: Path, rows: pd.DataFrame) -> dict[str, torch.Tensor]:
    features = {}
    for uid in rows.seriesuid:
        path = feature_dir / f"{uid}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen feature {path}; run extraction first")
        record = torch.load(path, map_location="cpu", weights_only=True)
        features[uid] = record["feature"].float()
    return features


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    prediction = (scores >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan"),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
    }


def cosine_warmup(epoch: int, warmup: int, epochs: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = min(max((epoch - warmup) / max(1, epochs - warmup), 0), 1)
    return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2


def fit_fold(args: argparse.Namespace, variant: str, fold: int, features: dict[str, torch.Tensor]) -> dict:
    rows = pd.read_csv(args.splits_dir / f"luna16_classification_fold{fold}.csv")
    rows = rows[rows.target.isin([0, 1])].copy()
    datasets = {}
    for split in ("train", "val", "test"):
        subset = rows[rows.split == split]
        x = torch.stack([features[uid] for uid in subset.seriesuid])
        y = torch.tensor(subset.target.to_numpy(), dtype=torch.float32)
        datasets[split] = (subset.seriesuid.tolist(), x, y)
    _train_ids, train_x, train_y = datasets["train"]
    counts = torch.bincount(train_y.long(), minlength=2).float()
    weights = (1 / counts.clamp_min(1))[train_y.long()]
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    sampler = WeightedRandomSampler(weights, len(train_y), replacement=True)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, sampler=sampler)
    head = nn.Linear(2 * EMBED_DIM, 1).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: cosine_warmup(epoch, args.warmup_epochs, args.epochs))
    criterion = nn.BCEWithLogitsLoss()
    out_dir = args.output_dir / variant / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_mcc, best_epoch, stale = -float("inf"), -1, 0
    history = []
    _val_ids, val_x, val_y = datasets["val"]
    for epoch in range(args.epochs):
        head.train()
        losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(x.to(args.device)).squeeze(1)
            loss = criterion(logits, y.to(args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        head.eval()
        with torch.inference_mode():
            val_scores = head(val_x.to(args.device)).sigmoid().squeeze(1).cpu().numpy()
        val_metrics = metrics(val_y.numpy(), val_scores)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["mcc"] > best_mcc:
            best_mcc, best_epoch, stale = val_metrics["mcc"], epoch, 0
            state = {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()}
            torch.save({"head": state, "epoch": epoch, "val_mcc": best_mcc, "val_auc": val_metrics["auc"]}, out_dir / "best_val_mcc.pt")
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    checkpoint = torch.load(out_dir / "best_val_mcc.pt", map_location=args.device, weights_only=True)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    test_ids, test_x, test_y = datasets["test"]
    with torch.inference_mode():
        scores = head(test_x.to(args.device)).sigmoid().squeeze(1).cpu().numpy()
    result = {"fold": fold, "variant": variant, "best_epoch": best_epoch, "best_val_mcc": best_mcc, **{f"test_{k}": v for k, v in metrics(test_y.numpy(), scores).items()}}
    pd.DataFrame({"fold": fold, "variant": variant, "seriesuid": test_ids, "label": test_y.long().numpy(), "score": scores, "prediction": (scores >= .5).astype(int)}).to_csv(out_dir / "test_predictions.csv", index=False)
    (out_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    args = args_parser()
    if args.epochs < 1 or args.batch_size < 1 or args.micro_batch_windows < 1:
        raise ValueError("epochs, batch-size and micro-batch-windows must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num-shards >= 1 and 0 <= shard-index < num-shards")
    if args.num_shards != 1 and not args.extract_only:
        raise ValueError("Sharded extraction requires --extract-only; train after all shards finish")
    if args.device == "cuda" and not torch.cuda.is_available() and not args.skip_extraction:
        raise RuntimeError("CUDA unavailable: cannot extract frozen 3DINO features")
    rows = classification_rows(args.splits_dir)
    variants = ("whole", "detections") if args.variant == "both" else (args.variant,)
    candidates = detector_candidates(args.detector_csv, set(rows.seriesuid)) if "detections" in variants else None
    if candidates is not None:
        validate_detector_oof_folds(candidates, args.splits_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        variant_dir = args.output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "config.json").write_text(json.dumps({**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "representation": "frozen CLS mean+max; linear 2048-to-1 head", "checkpoint_selection": "maximum validation MCC, threshold 0.5", "detection_policy": "top 4 with score >=0.50; otherwise highest-score candidate"}, indent=2))
        if not args.skip_extraction:
            extract_features(args, variant, rows, candidates)
        if args.extract_only:
            continue
        features = load_features(args.output_dir / variant / "features", rows)
        results = [fit_fold(args, variant, fold, features) for fold in args.folds]
        pd.DataFrame(results).to_csv(args.output_dir / variant / "fold_results.csv", index=False)
        if sorted(args.folds) == list(range(10)):
            predictions = pd.concat([
                pd.read_csv(args.output_dir / variant / f"fold_{fold}" / "test_predictions.csv", dtype={"seriesuid": str})
                for fold in args.folds
            ], ignore_index=True)
            if predictions.seriesuid.duplicated().any() or len(predictions) != len(rows):
                raise ValueError("Pooled predictions must contain each classified scan exactly once")
            predictions.to_csv(args.output_dir / variant / "oof_test_predictions.csv", index=False)
            pooled = metrics(predictions.label.to_numpy(), predictions.score.to_numpy())
            (args.output_dir / variant / "pooled_test_metrics.json").write_text(json.dumps({"n": len(predictions), **pooled}, indent=2))
            print(json.dumps({"variant": variant, "n": len(predictions), **pooled}), flush=True)


if __name__ == "__main__":
    main()
