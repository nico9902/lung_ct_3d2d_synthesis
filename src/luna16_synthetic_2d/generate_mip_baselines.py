from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    image = np.clip(image, 0.0, 255.0)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def resize_channel(channel: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    pil = Image.fromarray(normalize_uint8(channel), mode="L")
    pil = pil.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def baseline_image(volume: np.ndarray, mode: str, image_size: tuple[int, int]) -> Image.Image:
    if mode == "axial":
        axial = volume.max(axis=0)
        channel = resize_channel(axial, image_size)
        rgb = np.stack([channel, channel, channel], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    if mode == "triview":
        axial = resize_channel(volume.max(axis=0), image_size)
        coronal = resize_channel(volume.max(axis=1), image_size)
        sagittal = resize_channel(volume.max(axis=2), image_size)
        rgb = np.stack([axial, coronal, sagittal], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    if mode == "central_axial":
        axial = volume[volume.shape[0] // 2]
        channel = resize_channel(axial, image_size)
        rgb = np.stack([channel, channel, channel], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    if mode == "central_triview":
        axial = resize_channel(volume[volume.shape[0] // 2], image_size)
        coronal = resize_channel(volume[:, volume.shape[1] // 2, :], image_size)
        sagittal = resize_channel(volume[:, :, volume.shape[2] // 2], image_size)
        rgb = np.stack([axial, coronal, sagittal], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    raise ValueError(f"Unsupported non-adaptive baseline mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate non-adaptive LUNA16 2D baselines.")
    parser.add_argument("--data-root", default="/ssd2/domenico/datasets/LUNA16_preprocessed")
    parser.add_argument(
        "--split-csv",
        default="/ssd2/domenico/datasets/LUNA16_preprocessed/cv_splits/luna16_classification_fold0.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["axial", "triview", "central_axial", "central_triview"], required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 384], metavar=("H", "W"))
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.split_csv)
    if not args.include_uncertain:
        df = df[df["target"].isin([0, 1])].copy()
    df = df.drop_duplicates("seriesuid").reset_index(drop=True)

    rows = []
    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    for index, row in df.iterrows():
        seriesuid = str(row["seriesuid"])
        sample_dir = output_dir / seriesuid
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_path = sample_dir / f"surface_{seriesuid}.png"
        if output_path.exists() and not args.overwrite:
            rows.append({"seriesuid": seriesuid, "path": str(output_path), "status": "exists"})
            continue

        image_path = data_root / row["image_path"]
        image = sitk.ReadImage(str(image_path))
        volume = sitk.GetArrayFromImage(image).astype(np.float32)
        pil = baseline_image(volume, args.mode, image_size)
        pil.save(output_path)
        rows.append(
            {
                "seriesuid": seriesuid,
                "path": str(output_path),
                "status": "written",
                "source_shape_dhw": tuple(int(v) for v in volume.shape),
                "image_height": image_size[0],
                "image_width": image_size[1],
            }
        )
        if (index + 1) % 50 == 0:
            print(f"Generated {index + 1}/{len(df)} {args.mode} baseline images", flush=True)

    manifest = {
        "mode": args.mode,
        "data_root": str(data_root),
        "split_csv": str(args.split_csv),
        "output_dir": str(output_dir),
        "image_size": list(image_size),
        "n_samples": len(df),
        "include_uncertain": bool(args.include_uncertain),
    }
    with (output_dir / "baseline_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    pd.DataFrame(rows).to_csv(output_dir / "baseline_manifest.csv", index=False)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
