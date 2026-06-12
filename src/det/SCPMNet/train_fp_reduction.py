from __future__ import annotations

import argparse
from pathlib import Path
import sys
import wandb

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.det.SCPMNet.fp_reduction import CandidatePatchDataset, FPReductionLitModel


def make_loader(
    dataset: CandidatePatchDataset,
    batch_size: int,
    num_workers: int,
    balanced: bool,
    shuffle: bool,
    samples_per_epoch: int | None = None,
) -> DataLoader:
    sampler = None
    if balanced:
        labels = dataset.labels()
        class_counts = np.bincount(labels, minlength=2).astype(np.float32)
        class_counts[class_counts == 0] = 1.0
        weights = 1.0 / class_counts[labels]
        num_samples = int(samples_per_epoch) if samples_per_epoch else len(weights)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_logger(args: argparse.Namespace, extra_config: dict):
    if not args.use_wandb:
        return False

    from pytorch_lightning.loggers import WandbLogger
    wandb.login(key="5eb716fd87389d240533319f8751488e37103d23")
    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name or Path(args.output_dir).name,
        entity=args.wandb_entity,
        save_dir=args.wandb_save_dir,
        mode=args.wandb_mode,
        log_model=args.wandb_log_model,
        config={**vars(args), **extra_config},
    )
    logger.experiment.define_metric("trainer/global_step")
    logger.experiment.define_metric("*", step_metric="trainer/global_step")
    return logger


def finalize_logger(logger, status: str) -> None:
    if not logger:
        return
    logger.finalize(status)
    experiment = getattr(logger, "experiment", None)
    if experiment is not None and hasattr(experiment, "finish"):
        experiment.finish(exit_code=0 if status == "success" else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a second-stage SCPMNet false-positive reduction classifier.")
    parser.add_argument("--train-candidates", required=True)
    parser.add_argument("--val-candidates", required=True)
    parser.add_argument("--csv-path", default="data/lidc_process/lidc_labels.csv")
    parser.add_argument("--data-root", default="data/lidc_process")
    parser.add_argument("--output-dir", default="outputs/scpmnet/fp_reduction")
    parser.add_argument("--patch-size", type=int, nargs=3, default=(32, 32, 32))
    parser.add_argument("--clip", type=float, nargs=2, default=(-1000.0, 400.0))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--volume-cache-size", type=int, default=4, help="Per-worker number of CT volumes to keep in RAM.")
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Number of sampled training candidates per epoch when using the balanced sampler.",
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--precision", default="32")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument(
        "--pos-weight",
        default="balanced_only",
        help=(
            "Positive BCE weight. Use a float, 'auto' for negatives/positives, "
            "'none' to disable, or 'balanced_only' to auto-enable only with --no-balanced-sampler."
        ),
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lung_ct_3d2d_synthesis_detection")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-save-dir", default="wandb2")
    parser.add_argument("--wandb-log-model", action="store_true")
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CandidatePatchDataset(
        candidates_csv=args.train_candidates,
        csv_path=args.csv_path,
        split="train",
        data_root=args.data_root,
        patch_size=args.patch_size,
        clip=args.clip,
        augment=True,
        volume_cache_size=args.volume_cache_size,
    )
    val_ds = CandidatePatchDataset(
        candidates_csv=args.val_candidates,
        csv_path=args.csv_path,
        split="val",
        data_root=args.data_root,
        patch_size=args.patch_size,
        clip=args.clip,
        augment=False,
        volume_cache_size=args.volume_cache_size,
    )
    print(f"Train candidates: {len(train_ds)} | positives={int(train_ds.labels().sum())} negatives={int((train_ds.labels() == 0).sum())}")
    print(f"Val candidates: {len(val_ds)} | positives={int(val_ds.labels().sum())} negatives={int((val_ds.labels() == 0).sum())}")

    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balanced=not args.no_balanced_sampler,
        shuffle=True,
        samples_per_epoch=args.samples_per_epoch,
    )
    val_loader = make_loader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, balanced=False, shuffle=False)

    labels = train_ds.labels()
    neg = max(int((labels == 0).sum()), 1)
    pos = max(int((labels == 1).sum()), 1)
    if str(args.pos_weight).lower() == "auto":
        pos_weight = neg / pos
    elif str(args.pos_weight).lower() == "none":
        pos_weight = None
    elif str(args.pos_weight).lower() == "balanced_only":
        pos_weight = neg / pos if args.no_balanced_sampler else None
    else:
        pos_weight = float(args.pos_weight)
    if pos_weight is not None:
        print(f"Using BCE pos_weight={pos_weight:.6g}")
    elif not args.no_balanced_sampler:
        print("Using balanced sampler without BCE pos_weight.")
    else:
        print("Using unweighted BCE on natural candidate distribution.")
    model = FPReductionLitModel(lr=args.lr, weight_decay=args.weight_decay, pos_weight=pos_weight)
    logger = build_logger(
        args,
        {
            "train_candidates_count": len(train_ds),
            "train_positives": int(train_ds.labels().sum()),
            "train_negatives": int((train_ds.labels() == 0).sum()),
            "val_candidates_count": len(val_ds),
            "val_positives": int(val_ds.labels().sum()),
            "val_negatives": int((val_ds.labels() == 0).sum()),
            "effective_pos_weight": pos_weight,
            "balanced_sampler": not args.no_balanced_sampler,
        },
    )
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="epoch={epoch:03d}-val_loss={val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        default_root_dir=str(output_dir),
        callbacks=[checkpoint],
        logger=logger,
    )
    status = "success"
    try:
        trainer.fit(model, train_loader, val_loader)
        print(f"Best checkpoint: {checkpoint.best_model_path}")
    except Exception:
        status = "failed"
        raise
    finally:
        finalize_logger(logger, status)


if __name__ == "__main__":
    main()
