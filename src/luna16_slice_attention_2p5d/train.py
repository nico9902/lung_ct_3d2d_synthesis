from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from .datamodule import Luna16SliceAttentionDataModule
from .lightning_model import SliceAttentionLightningModule
from .model import SUPPORTED_BACKBONES


class CacheWriteFirstEpochOnly(Callback):
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or trainer.current_epoch != 0:
            return
        trainer.datamodule.disable_cache_writes()
        print(
            "Cache writes disabled after epoch 0; later cache misses will not be persisted.",
            flush=True,
        )


def parse_devices(devices: str):
    if devices == "auto":
        return "auto"
    value = devices.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    if "," in value:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Lightning 2.5D LUNA16 classifier with gated axial-slice attention."
    )
    parser.add_argument("--data-root", default="/ssd2/domenico/datasets/LUNA16_preprocessed")
    parser.add_argument(
        "--splits-dir", default="/ssd2/domenico/datasets/LUNA16_preprocessed/cv_splits"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/luna16_slice_attention_2p5d_all_slices_256x384_effnetv2s",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, default="efficientnet_v2_s")
    parser.add_argument("--image-size", type=int, nargs="+", default=[256, 384])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--slice-chunk-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument(
        "--precision",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        default="bf16-mixed",
    )
    parser.add_argument("--monitor", choices=["val_auc", "val_mcc", "val_loss"], default="val_mcc")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument("--freeze-backbone", action="store_true")
    freeze_group.add_argument(
        "--freeze-half-backbone",
        action="store_true",
        help="Freeze approximately the first 50%% of backbone weights by parameter count.",
    )
    parser.add_argument("--cache-dir", default=None)
    cache_write_group = parser.add_mutually_exclusive_group()
    cache_write_group.add_argument(
        "--cache-read-only",
        action="store_true",
        help="Load existing cached tensors but do not write missing cache files.",
    )
    cache_write_group.add_argument(
        "--cache-write-first-epoch-only",
        action="store_true",
        help="Write missing cache entries during epoch 0, then make the cache read-only.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="luna16-slice-attention-2p5d")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="efficientnet_v2_s_all_slices_256x384")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--limit-test-samples", type=int, default=None)
    return parser.parse_args()


def build_loggers(args: argparse.Namespace, output_dir: Path):
    loggers = [CSVLogger(save_dir=str(output_dir), name="lightning")]
    if not args.wandb:
        return loggers

    from pytorch_lightning.loggers import WandbLogger

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    loggers.append(
        WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or f"fold_{args.fold}_{args.backbone}_slice_attention",
            group=args.wandb_group,
            tags=["luna16", "2.5d", "slice-attention", args.backbone, f"fold_{args.fold}"],
            save_dir=str(output_dir),
            offline=args.wandb_offline,
            log_model=False,
        )
    )
    return loggers


def save_test_metrics(test_results: list[dict[str, object]], output_dir: Path) -> None:
    metrics = dict(test_results[0]) if test_results else {}
    serializable = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        serializable[key] = float(value) if isinstance(value, (int, float)) else value
    with (output_dir / "test_metrics.json").open("w") as handle:
        json.dump(serializable, handle, indent=2)


def save_test_predictions(
    model: SliceAttentionLightningModule,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    predictions = pd.DataFrame(model.test_prediction_rows)
    if predictions.empty:
        return
    predictions.insert(0, "backbone", args.backbone)
    predictions.insert(0, "fold", args.fold)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed + args.fold, workers=True)

    output_dir = Path(args.output_dir) / f"fold_{args.fold}" / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    datamodule = Luna16SliceAttentionDataModule(
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        fold=args.fold,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        cache_read_only=args.cache_read_only,
        limit_train_samples=args.limit_train_samples,
        limit_val_samples=args.limit_val_samples,
        limit_test_samples=args.limit_test_samples,
    )
    if args.cache_write_first_epoch_only:
        datamodule.setup("fit")
        expected_cache_entries, created_cache_entries = (
            datamodule.complete_missing_compact_cache()
        )
        print(
            f"Compact cache ready: {expected_cache_entries}/{expected_cache_entries}; "
            f"created {created_cache_entries} missing entries before epoch 0.",
            flush=True,
        )

    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="{epoch:03d}-{val_loss:.4f}-{val_auc:.4f}-{val_mcc:.4f}",
            monitor=args.monitor,
            mode="min" if args.monitor == "val_loss" else "max",
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if not args.no_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=args.monitor,
                mode="min" if args.monitor == "val_loss" else "max",
                patience=args.patience,
            )
        )
    if args.cache_write_first_epoch_only:
        callbacks.append(CacheWriteFirstEpochOnly())

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        precision=args.precision,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=10.0,
        logger=build_loggers(args, output_dir),
        callbacks=callbacks,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        benchmark=True,
    )

    model = SliceAttentionLightningModule(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        slice_chunk_size=args.slice_chunk_size,
        freeze_backbone=args.freeze_backbone,
        freeze_half_backbone=args.freeze_half_backbone,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
    )
    for logger in trainer.loggers:
        logger.log_hyperparams(vars(args))

    trainer.fit(model=model, datamodule=datamodule)
    test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
    save_test_metrics(test_results, output_dir)
    save_test_predictions(model, output_dir, args)


if __name__ == "__main__":
    main()
