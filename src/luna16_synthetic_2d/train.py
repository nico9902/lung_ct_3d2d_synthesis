from __future__ import annotations

import argparse
import json
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
        "--manifest-csv",
        default="outputs/scpmnet_luna16_10fold_tps_images/manifest.csv",
        help="TPS manifest CSV with synthetic_image paths.",
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
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--monitor", default="val_auc", choices=["val_auc", "val_loss", "val_acc"])
    parser.add_argument("--eval-only", default=None, help="Lightning .ckpt path to evaluate.")
    return parser.parse_args()


def parse_devices(devices: str):
    if devices == "auto":
        return "auto"
    if "," in devices:
        return [int(item) for item in devices.split(",")]
    return int(devices)


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir) / f"fold_{args.fold}" / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    datamodule = SyntheticLuna16DataModule(
        manifest_csv=args.manifest_csv,
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
    logger = CSVLogger(save_dir=output_dir.parent, name=args.backbone)

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        precision=args.precision,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=10,
    )

    if args.eval_only:
        model = SyntheticLuna16Classifier.load_from_checkpoint(args.eval_only, pretrained=False)
        trainer.test(model=model, datamodule=datamodule)
        return

    model = SyntheticLuna16Classifier(
        backbone=args.backbone,
        num_classes=len(args.classes),
        class_names=args.classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        max_epochs=args.epochs,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
