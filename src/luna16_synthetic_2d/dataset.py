from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


ARRAY_EXTENSIONS = {".npy"}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int
    class_name: str
    split: str
    sample_id: str


def _normalise_class_name(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {
        "0": "benign",
        "1": "malignant",
        "benign": "benign",
        "normal": "benign",
        "negative": "benign",
        "malignant": "malignant",
        "positive": "malignant",
        "cancer": "malignant",
    }
    return aliases.get(value, value)


def _label_column(df: pd.DataFrame, csv_path: Path) -> str:
    if "target_name" in df.columns:
        return "target_name"
    if "target" in df.columns:
        return "target"
    if "label" in df.columns:
        return "label"
    raise ValueError(f"{csv_path} must contain one of: target_name, target, label")


def _read_samples(
    manifest_csv: str | Path,
    split_csv: str | Path,
    split: str,
    classes: Iterable[str],
    image_column: str,
) -> list[ImageSample]:
    manifest_csv = Path(manifest_csv)
    split_csv = Path(split_csv)
    manifest = pd.read_csv(manifest_csv)
    split_df = pd.read_csv(split_csv)
    path_search_roots = [Path.cwd(), *manifest_csv.parents, *split_csv.parents]

    for path, df, required in [
        (manifest_csv, manifest, ["seriesuid", image_column]),
        (split_csv, split_df, ["seriesuid", "split"]),
    ]:
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

    label_col = _label_column(split_df, split_csv)
    split_df = split_df[split_df["split"].astype(str) == split].copy()
    merged = split_df.merge(manifest[["seriesuid", image_column]], on="seriesuid", how="inner")

    normalised_classes = [_normalise_class_name(c) for c in classes]
    class_to_idx = {name: idx for idx, name in enumerate(normalised_classes)}
    samples: list[ImageSample] = []
    missing_files = 0

    for _, row in merged.iterrows():
        class_name = _normalise_class_name(row[label_col])
        if class_name not in class_to_idx:
            continue

        image_path = _resolve_image_path(row[image_column], path_search_roots)
        if not image_path.exists():
            missing_files += 1
            continue

        samples.append(
            ImageSample(
                path=image_path,
                label=class_to_idx[class_name],
                class_name=class_name,
                split=split,
                sample_id=str(row["seriesuid"]),
            )
        )

    if not samples:
        raise RuntimeError(
            f"No images found for split '{split}' after joining {manifest_csv} with {split_csv}. "
            f"Skipped missing files: {missing_files}."
        )
    return samples


def _resolve_image_path(path_value, search_roots: Iterable[Path]) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    for root in search_roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    return path


class SyntheticLuna16Dataset(Dataset):
    """LUNA16 synthetic image dataset driven by a fold classification CSV."""

    def __init__(
        self,
        manifest_csv: str | Path,
        split_csv: str | Path,
        split: str,
        transform=None,
        classes: Iterable[str] = ("benign", "malignant"),
        image_column: str = "synthetic_image",
    ) -> None:
        self.samples = _read_samples(
            manifest_csv=manifest_csv,
            split_csv=split_csv,
            split=split,
            classes=classes,
            image_column=image_column,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = load_image(sample.path)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(sample.label, dtype=torch.long)
        return image, label, sample.sample_id


def load_image(path: str | Path) -> Image.Image:
    path = Path(path)
    if path.suffix.lower() in ARRAY_EXTENSIONS:
        array = np.load(path).astype(np.float32)
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        array = np.nan_to_num(array)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D synthetic image array, got {array.shape} for {path}")
        min_value = float(array.min())
        max_value = float(array.max())
        if 0.0 <= min_value and max_value <= 1.0:
            array = array * 255.0
        elif max_value > min_value:
            array = (array - min_value) / (max_value - min_value) * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    return Image.open(path).convert("RGB")
