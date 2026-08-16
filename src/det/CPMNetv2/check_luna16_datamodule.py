from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.det.CPMNetv2.luna16_datamodule import Luna16CPMNetDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test CPMNetv2 LUNA16 datamodule reads.")
    parser.add_argument("--data-root", default="/ssd2/domenico/datasets/LUNA16_preprocessed")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--crop-size", nargs=3, type=int, default=(96, 96, 96), metavar=("D", "H", "W"))
    parser.add_argument("--overlap-size", nargs=3, type=int, default=(24, 24, 24), metavar=("D", "H", "W"))
    parser.add_argument("--spacing", nargs=3, type=float, default=(1.0, 1.0, 1.0), metavar=("Z", "Y", "X"))
    parser.add_argument("--val-full-volume", action="store_true")
    return parser.parse_args()


def describe_crop_batch(batch):
    image = batch["image"]
    annot = batch["annot"]
    foreground = annot[..., -1] >= 0
    print("train_batch_image_shape", tuple(image.shape))
    print("train_batch_image_minmax", float(image.min()), float(image.max()))
    print("train_batch_annot_shape", tuple(annot.shape))
    print("train_batch_foreground_boxes", int(foreground.sum().item()))
    if foreground.any():
        selected = annot[foreground][0].detach().cpu().numpy()
        print("first_foreground_zyxdhwc", np.array2string(selected, precision=2))


def describe_eval_batch(batch):
    split_images = batch["split_images"]
    print("eval_file_name", batch["file_name"][0])
    print("eval_split_images_shape", tuple(split_images.shape))
    print("eval_split_images_minmax", float(split_images.min()), float(split_images.max()))
    print("eval_nzhw", batch["nzhw"])
    print("eval_spacing", batch["spacing"])


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    csv_path = Path(args.csv_path) if args.csv_path else data_root / "cv_splits" / f"luna16_fold{args.fold}.csv"

    dm = Luna16CPMNetDataModule.from_split_csv(
        csv_path=str(csv_path),
        images_dir=str(data_root),
        labels_csv=str(csv_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
        overlap_size=args.overlap_size,
        spacing=args.spacing,
        num_samples=args.num_samples,
        val_full_volume=args.val_full_volume,
    )
    dm.setup("fit")
    dm.setup("test")

    print("csv_path", csv_path)
    print("train_cases", len(dm.train_cases))
    print("val_cases", len(dm.val_cases))
    print("test_cases", len(dm.test_cases))
    print("train_dataset_len", len(dm.train_ds))
    print("val_dataset_len", len(dm.val_ds))
    print("test_dataset_len", len(dm.test_ds))

    train_batch = next(iter(dm.train_dataloader()))
    describe_crop_batch(train_batch)

    if args.val_full_volume:
        val_batch = next(iter(dm.val_dataloader()))
        describe_eval_batch(val_batch)
    test_batch = next(iter(dm.test_dataloader()))
    describe_eval_batch(test_batch)

    annotation_df, seriesuid_df = dm.test_ds.annotations_dataframe()
    print("test_annotation_rows", len(annotation_df))
    print("test_seriesuid_rows", len(seriesuid_df))
    print("first_test_annotation", annotation_df.head(1).to_dict(orient="records"))


if __name__ == "__main__":
    main()
