from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.luna16_slice_attention_2p5d.datamodule import SliceAttentionLuna16DataModule
from src.luna16_slice_attention_2p5d.lightning_model import SliceAttentionLightningModule
from src.luna16_synthetic_2d.models import SUPPORTED_BACKBONES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LUNA16 2.5D slice-attention patient classifier.")
    parser.add_argument("--output-dir", default="outputs/luna16_slice_attention_2p5d")
    parser.add_argument("--data-root", default="/ssd2/domenico/datasets/LUNA16_preprocessed")
    parser.add_argument("--split-csv", default=None)
    parser.add_argument("--splits-dir", default="/ssd2/domenico/datasets/LUNA16_preprocessed/cv_splits")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--backbone", default="efficientnet_v2_s", choices=SUPPORTED_BACKBONES)
    parser.add_argument("--classes", nargs="+", default=["benign", "malignant"])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 384], metavar=("H", "W"))
    parser.add_argument("--encoder-chunk-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument("--monitor", default="val_mcc", choices=["val_auc", "val_loss", "val_acc", "val_f1", "val_mcc"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="luna16-slice-attention-2p5d")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-log-model", action="store_true")
    parser.add_argument("--eval-only", default=None)
    return parser.parse_args()


def parse_devices(devices: str, accelerator: str):
    if devices == "auto":
        return "auto"
    devices = devices.strip()
    is_list_syntax = devices.startswith("[") and devices.endswith("]")
    if is_list_syntax:
        devices = devices[1:-1].strip()
    if "," in devices:
        return [int(item.strip()) for item in devices.split(",") if item.strip()]
    if is_list_syntax and devices:
        return [int(devices)]
    if accelerator in ("gpu", "cuda") and devices.isdigit() and int(devices) == 0:
        return [0]
    return int(devices)


def build_loggers(args: argparse.Namespace, output_dir: Path):
    loggers = [CSVLogger(save_dir=output_dir.parent, name=args.backbone)]
    if args.wandb:
        import wandb
        from pytorch_lightning.loggers import WandbLogger

        name = args.wandb_name or f"fold_{args.fold}_{args.backbone}_slice_attention"
        group = args.wandb_group or f"{args.backbone}_slice_attention_2p5d"
        os.environ["WANDB_RUN_GROUP"] = group
        os.environ["WANDB_NAME"] = name
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        else:
            wandb.login(key="5eb716fd87389d240533319f8751488e37103d23")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=name,
            group=group,
            job_type=f"fold_{args.fold}",
            tags=[args.backbone, f"fold_{args.fold}", "slice_attention_2p5d"],
            dir=str(output_dir.parent),
            config=vars(args),
            reinit=True,
        )
        loggers.append(WandbLogger(experiment=run, offline=args.wandb_offline, log_model=args.wandb_log_model))
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


def save_test_predictions(model: SliceAttentionLightningModule, output_dir: Path, args: argparse.Namespace) -> None:
    rows = getattr(model, "test_prediction_rows", [])
    if not rows:
        return
    predictions = pd.DataFrame(rows)
    predictions.insert(0, "backbone", args.backbone)
    predictions.insert(0, "fold", args.fold)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir) / f"fold_{args.fold}" / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    datamodule = SliceAttentionLuna16DataModule(
        data_root=args.data_root,
        split_csv=args.split_csv,
        splits_dir=args.splits_dir,
        fold=args.fold,
        classes=args.classes,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if not args.eval_only:
        datamodule.setup("fit")
        args.class_counts = datamodule.class_counts.tolist() if datamodule.class_counts is not None else None
        with (output_dir / "config.json").open("w") as handle:
            json.dump(vars(args), handle, indent=2)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="{epoch:03d}-{val_loss:.4f}-{val_auc:.4f}-{val_mcc:.4f}",
        monitor=args.monitor,
        mode="min" if args.monitor == "val_loss" else "max",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices, args.accelerator),
        precision=args.precision,
        max_epochs=args.epochs,
        logger=build_loggers(args, output_dir),
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=10,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_sanity_val_steps=0,
    )

    if args.eval_only:
        model = SliceAttentionLightningModule.load_from_checkpoint(args.eval_only, pretrained=False)
        test_results = trainer.test(model=model, datamodule=datamodule)
        save_test_metrics(test_results, output_dir)
        save_test_predictions(model, output_dir, args)
        return

    model = SliceAttentionLightningModule(
        backbone=args.backbone,
        num_classes=len(args.classes),
        class_names=args.classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
        max_epochs=args.epochs,
        encoder_chunk_size=args.encoder_chunk_size,
    )
    for logger in trainer.loggers:
        logger.log_hyperparams(vars(args))
    trainer.fit(model=model, datamodule=datamodule)
    test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
    save_test_metrics(test_results, output_dir)
    save_test_predictions(model, output_dir, args)


if __name__ == "__main__":
    main()
