"""Train/evaluate the partial-fine-tuning 3DINO-ViT baseline on one LUNA16 fold.

Uses the exact same fold CSV (``luna16_classification_fold{fold}.csv``)
already used by the Adaptive RBF and Rad-JEPA experiments -- no new folds are
created. One wandb run per fold (``3dino_partial_ft_fold{N}``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import DeviceStatsMonitor, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.luna16_3dino_3d.datamodule import Luna163DinoDataModule  # noqa: E402
from src.luna16_3dino_3d.lightning_model import Dino3DPartialFinetuneClassifier  # noqa: E402
from src.luna16_3dino_3d.model import NATIVE_INPUT_SIZE  # noqa: E402


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Partial-fine-tuning 3DINO-ViT LUNA16 baseline (one fold).")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--splits-dir", default="data/processed/cv_splits")
    parser.add_argument("--output-dir", default="outputs/luna16_3dino_3d")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint-path", default=None, help="Path to the official 3DINO-ViT teacher_checkpoint (.pth).")

    parser.add_argument("--window-size", type=int, default=NATIVE_INPUT_SIZE)
    parser.add_argument("--stride", type=int, default=NATIVE_INPUT_SIZE // 2, help="~50%% overlap by default.")
    parser.add_argument("--max-windows", type=int, default=None, help="Cap windows/patient (debugging/memory).")
    parser.add_argument(
        "--windows-cache-dir",
        default="outputs/luna16_3dino_3d/window_cache",
        help="Cache preprocessed per-patient window tensors here (shared across all 10 folds).",
    )
    parser.add_argument("--micro-batch-windows", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--accumulate-grad-batches", type=int, default=8, help="Patients per effective batch.")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--precision", default="bf16-true")
    parser.add_argument("--monitor", default="val_auc", choices=["val_auc", "val_mcc", "val_loss"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="luna16-3dino-3d")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-offline", action="store_true")
    return parser.parse_args()


def build_loggers(args: argparse.Namespace, output_dir: Path, extra_config: dict) -> list:
    loggers = [CSVLogger(save_dir=str(output_dir.parent), name=f"fold_{args.fold}")]
    if args.wandb:
        import wandb
        from pytorch_lightning.loggers import WandbLogger

        name = f"3dino_partial_ft_fold{args.fold}"
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        else:
            wandb.login()  # uses WANDB_API_KEY env var or ~/.netrc; no hardcoded credentials
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=name,
            group="3dino_partial_ft",
            job_type=f"fold_{args.fold}",
            tags=["3dino", "partial-finetune", f"fold_{args.fold}"],
            dir=str(output_dir.parent),
            config={**vars(args), **extra_config},
            reinit=True,
        )
        loggers.append(WandbLogger(experiment=run, offline=args.wandb_offline))
    return loggers


def save_test_outputs(model: Dino3DPartialFinetuneClassifier, output_dir: Path, fold: int) -> dict:
    rows = model.test_prediction_rows
    with (output_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "representation", "sample_id", "split", "label", "prediction", "score"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "fold": fold,
                    "representation": "3dino_partial_ft",
                    "sample_id": row["sample_id"],
                    "split": "test",
                    "label": row["label"],
                    "prediction": row["prediction"],
                    "score": row["score"],
                }
            )

    labels = [r["label"] for r in rows]
    preds = [r["prediction"] for r in rows]
    scores = [r["score"] for r in rows]
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics = {
        "test_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan"),
        "test_mcc": float(matthews_corrcoef(labels, preds)),
        "test_acc": float(accuracy_score(labels, preds)),
        "test_f1": float(f1_score(labels, preds, zero_division=0)),
        "test_precision": float(precision_score(labels, preds, zero_division=0)),
        "test_recall": float(recall_score(labels, preds, zero_division=0)),
        "test_tn": int(tn),
        "test_fp": int(fp),
        "test_fn": int(fn),
        "test_tp": int(tp),
    }
    with (output_dir / "test_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed + args.fold, workers=True)

    output_dir = Path(args.output_dir) / f"fold_{args.fold}" / "3dino_partial_ft"
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = Luna163DinoDataModule(
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        fold=args.fold,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
        windows_cache_dir=args.windows_cache_dir,
        num_workers=args.num_workers,
    )

    model = Dino3DPartialFinetuneClassifier(
        checkpoint_path=args.checkpoint_path,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        micro_batch_windows=args.micro_batch_windows,
    )

    config = {
        "checkpoint_path": args.checkpoint_path,
        "batch_size": 1,  # one patient per step; all its windows are batched internally
        "effective_batch_size": args.accumulate_grad_batches,
        "num_frozen_blocks": len(model.freeze_report["frozen_blocks"]),
        "num_trainable_blocks": len(model.freeze_report["trainable_blocks"]),
        "total_params": model.freeze_report["total_params"],
        "trainable_params": model.freeze_report["trainable_params"],
        "trainable_pct": model.freeze_report["trainable_pct"],
        "window_size": args.window_size,
        "stride": args.stride,
        "preprocessing": "percentile[0.05,99.95]->[-1,1] per window (official 3DINO demo convention)",
        "torch_version": torch.__version__,
        "pytorch_lightning_version": pl.__version__,
        "hostname": platform.node(),
        "git_commit": git_commit(),
    }
    with (output_dir / "config.json").open("w") as handle:
        json.dump({**vars(args), **config}, handle, indent=2)
    with (output_dir / "freeze_report.json").open("w") as handle:
        json.dump(model.freeze_report, handle, indent=2)

    loggers = build_loggers(args, output_dir, config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="{epoch:03d}-{val_auc:.4f}-{val_mcc:.4f}",
        monitor=args.monitor,
        mode="min" if args.monitor == "val_loss" else "max",
        save_top_k=1,
        save_weights_only=True,
    )
    early_stop = EarlyStopping(
        monitor=args.monitor,
        mode="min" if args.monitor == "val_loss" else "max",
        patience=args.patience,
    )

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.epochs,
        logger=loggers,
        callbacks=[
            checkpoint_callback,
            early_stop,
            LearningRateMonitor(logging_interval="epoch"),
            *([DeviceStatsMonitor()] if torch.cuda.is_available() else []),
        ],
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=10,
        deterministic=False,  # 3D conv/attention kernels: exact determinism not guaranteed, seed still fixed
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    metrics = save_test_outputs(model, output_dir, args.fold)
    print(json.dumps({"fold": args.fold, **metrics}), flush=True)

    if args.wandb:
        import wandb

        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
