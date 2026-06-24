from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.luna16_synthetic_2d.datamodule import SyntheticLuna16DataModule
    from src.luna16_synthetic_2d.lightning_model import SyntheticLuna16Classifier
    from src.luna16_synthetic_2d.models import SUPPORTED_BACKBONES
else:
    from .datamodule import SyntheticLuna16DataModule
    from .lightning_model import SyntheticLuna16Classifier
    from .models import SUPPORTED_BACKBONES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch Lightning 2D classifier on LUNA16 synthetic nodule images."
    )
    parser.add_argument("--output-dir", default="outputs/luna16_synthetic_2d")
    parser.add_argument(
        "--synthetic-images-dir",
        default="outputs/luna16_saliency_synthetic_gt",
        help=(
            "Root directory containing synthetic images. Supports saliency GT "
            "<seriesuid>/surface_grid_float_<seriesuid>.npy and TPS "
            "fold_<fold>/<seriesuid>_tps_top5.npy layouts."
        ),
    )
    parser.add_argument("--split-csv", default=None, help="Optional explicit fold classification CSV.")
    parser.add_argument("--splits-dir", default="data/LUNA16_preprocessed/cv_splits")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--backbone", default="resnet18", choices=SUPPORTED_BACKBONES)
    parser.add_argument("--classes", nargs="+", default=["benign", "malignant"])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=[256, 384],
        help="Resize size as H W, or one value for square resize.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--accelerator", default="auto", help="Lightning accelerator: auto, cpu, gpu, mps.")
    parser.add_argument("--devices", default="auto", help="Lightning devices, e.g. auto, 1, 0, or 0,1.")
    parser.add_argument("--precision", default="32-true", help="Lightning precision, e.g. 32-true or 16-mixed.")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Lightning accumulate_grad_batches.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="luna16-synthetic-2d")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-log-model", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-half-backbone", action="store_true")
    parser.add_argument(
        "--freeze-first-layers",
        type=int,
        default=0,
        help="Freeze the first N non-head parameter groups.",
    )
    parser.add_argument(
        "--unfreeze-last-layers",
        type=int,
        default=0,
        help="Freeze the backbone and unfreeze only the last N non-head backbone parameter groups.",
    )
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--monitor", default="val_mcc", choices=["val_auc", "val_loss", "val_acc", "val_f1", "val_mcc"])
    parser.add_argument("--eval-only", default=None, help="Lightning .ckpt path to evaluate.")
    return parser.parse_args()


def parse_devices(devices: str, accelerator: str):
    if devices == "auto":
        return "auto"
    devices = devices.strip()
    is_list_syntax = devices.startswith("[") and devices.endswith("]")
    if devices.startswith("[") and devices.endswith("]"):
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

        name = args.wandb_name or f"fold_{args.fold}_{args.backbone}"
        group = args.wandb_group or args.backbone
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
            tags=[args.backbone, f"fold_{args.fold}"],
            dir=str(output_dir.parent),
            config=vars(args),
            reinit=True,
        )
        print(
            f"W&B run initialized: project={args.wandb_project}, "
            f"group={group}, name={name}, id={run.id}"
        )
        loggers.append(
            WandbLogger(
                experiment=run,
                offline=args.wandb_offline,
                log_model=args.wandb_log_model,
            )
        )
    return loggers


def save_test_metrics(test_results: list[dict[str, object]], output_dir: Path) -> None:
    metrics = dict(test_results[0]) if test_results else {}
    serializable_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        serializable_metrics[key] = float(value) if isinstance(value, (int, float)) else value

    with (output_dir / "test_metrics.json").open("w") as handle:
        json.dump(serializable_metrics, handle, indent=2)


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir) / f"fold_{args.fold}" / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    datamodule = SyntheticLuna16DataModule(
        synthetic_images_dir=args.synthetic_images_dir,
        split_csv=args.split_csv,
        splits_dir=args.splits_dir,
        fold=args.fold,
        classes=args.classes,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        image_size=args.image_size,
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
        filename="{epoch:03d}-{val_loss:.4f}-{val_auc:.4f}",
        monitor=args.monitor,
        mode="min" if args.monitor == "val_loss" else "max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor=args.monitor,
        mode="min" if args.monitor == "val_loss" else "max",
        patience=args.patience,
    )
    loggers = build_loggers(args, output_dir)

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices, args.accelerator),
        precision=args.precision,
        max_epochs=args.epochs,
        logger=loggers,
        callbacks=[checkpoint_callback, early_stop_callback, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=10,
        accumulate_grad_batches=args.accumulate_grad_batches if hasattr(args, "accumulate_grad_batches") else 1,
    )

    if args.eval_only:
        model = SyntheticLuna16Classifier.load_from_checkpoint(args.eval_only, pretrained=False)
        test_results = trainer.test(model=model, datamodule=datamodule)
        save_test_metrics(test_results, output_dir)
        return

    model = SyntheticLuna16Classifier(
        backbone=args.backbone,
        num_classes=len(args.classes),
        class_names=args.classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        freeze_half_backbone=args.freeze_half_backbone,
        freeze_first_layers=args.freeze_first_layers,
        unfreeze_last_layers=args.unfreeze_last_layers,
        max_epochs=args.epochs,
    )
    for logger in loggers:
        logger.log_hyperparams(vars(args))
    trainer.fit(model=model, datamodule=datamodule)
    test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
    save_test_metrics(test_results, output_dir)


if __name__ == "__main__":
    main()
