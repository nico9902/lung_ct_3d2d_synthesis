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
    sample_weight: float = 1.0


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
    synthetic_images_dir: str | Path,
    split_csv: str | Path,
    fold: int,
    split: str,
    classes: Iterable[str],
    image_suffix: str,
) -> list[ImageSample]:
    synthetic_images_dir = Path(synthetic_images_dir)
    split_csv = Path(split_csv)
    if not synthetic_images_dir.exists():
        raise FileNotFoundError(f"Synthetic images directory does not exist: {synthetic_images_dir}")
    split_df = pd.read_csv(split_csv)
    path_search_roots = [Path.cwd(), synthetic_images_dir, *synthetic_images_dir.parents, *split_csv.parents]

    for path, df, required in [
        (split_csv, split_df, ["seriesuid", "split"]),
    ]:
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

    label_col = _label_column(split_df, split_csv)
    split_df = split_df[split_df["split"].astype(str) == split].copy()

    normalised_classes = [_normalise_class_name(c) for c in classes]
    class_to_idx = {name: idx for idx, name in enumerate(normalised_classes)}
    samples: list[ImageSample] = []
    missing_files = 0
    first_missing_candidates: list[Path] | None = None

    for _, row in split_df.iterrows():
        class_name = _normalise_class_name(row[label_col])
        if class_name not in class_to_idx:
            continue

        sample_id = str(row["seriesuid"])
        image_path = _resolve_sample_image_path(
            synthetic_images_dir=synthetic_images_dir,
            fold=fold,
            sample_id=sample_id,
            image_suffix=image_suffix,
            search_roots=path_search_roots,
        )
        if not image_path.exists():
            missing_files += 1
            if first_missing_candidates is None:
                first_missing_candidates = _candidate_image_paths(
                    synthetic_images_dir=synthetic_images_dir,
                    fold=fold,
                    sample_id=sample_id,
                    image_suffix=image_suffix,
                )
            continue

        samples.append(
            ImageSample(
                path=image_path,
                label=class_to_idx[class_name],
                class_name=class_name,
                split=split,
                sample_id=sample_id,
                sample_weight=float(row.get("sample_weight", 1.0)),
            )
        )

    if not samples:
        candidate_hint = ""
        if first_missing_candidates is not None:
            formatted_candidates = ", ".join(str(path) for path in first_missing_candidates[:5])
            candidate_hint = f" First candidates tried: {formatted_candidates}."
        raise RuntimeError(
            f"No images found for split '{split}' in {synthetic_images_dir} using {split_csv}. "
            f"Skipped missing files: {missing_files}.{candidate_hint}"
        )
    return samples


def _candidate_image_paths(
    synthetic_images_dir: Path,
    fold: int,
    sample_id: str,
    image_suffix: str,
) -> list[Path]:
    fold_dir = synthetic_images_dir / f"fold_{fold}"
    return [
        fold_dir / f"{sample_id}{image_suffix}",
        synthetic_images_dir / f"{sample_id}{image_suffix}",
        fold_dir / sample_id / f"surface_{sample_id}.png",
        fold_dir / sample_id / f"surface_grid_float_{sample_id}.npy",
        fold_dir / sample_id / f"surface_grid_int_{sample_id}.npy",
        synthetic_images_dir / sample_id / f"surface_{sample_id}.png",
        synthetic_images_dir / sample_id / f"surface_grid_float_{sample_id}.npy",
        synthetic_images_dir / sample_id / f"surface_grid_int_{sample_id}.npy",
    ]


def _resolve_sample_image_path(
    synthetic_images_dir: Path,
    fold: int,
    sample_id: str,
    image_suffix: str,
    search_roots: Iterable[Path],
) -> Path:
    candidates = _candidate_image_paths(
        synthetic_images_dir=synthetic_images_dir,
        fold=fold,
        sample_id=sample_id,
        image_suffix=image_suffix,
    )
    for candidate in candidates:
        resolved = _resolve_image_path(candidate, search_roots)
        if resolved.exists():
            return resolved
    return candidates[0]


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
        synthetic_images_dir: str | Path,
        split_csv: str | Path,
        fold: int,
        split: str,
        transform=None,
        classes: Iterable[str] = ("benign", "malignant"),
        image_suffix: str = "_tps_top5.npy",
    ) -> None:
        self.samples = _read_samples(
            synthetic_images_dir=synthetic_images_dir,
            split_csv=split_csv,
            fold=fold,
            split=split,
            classes=classes,
            image_suffix=image_suffix,
        )
        self.labels = [sample.label for sample in self.samples]
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

    def get_sampler(self):
        labels = np.array(self.labels)
        neg_class_count = (labels == 0).sum()
        pos_class_count = (labels == 1).sum()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[sample.label] * sample.sample_weight for sample in self.samples]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        return sampler


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
