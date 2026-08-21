from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


@dataclass(frozen=True)
class VolumeSample:
    seriesuid: str
    image_path: Path
    label: int
    class_name: str
    split: str
    sample_weight: float = 1.0


class MinMaxScale:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        min_value = tensor.amin()
        max_value = tensor.amax()
        return (tensor - min_value) / (max_value - min_value).clamp_min(1e-6)


def build_slice_transform(image_size: tuple[int, int], train: bool):
    steps = [transforms.Resize(image_size)]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
            ]
        )
    steps.extend([transforms.ToTensor(), MinMaxScale()])
    return transforms.Compose(steps)


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
    for column in ("target_name", "target", "label"):
        if column in df.columns:
            return column
    raise ValueError(f"{csv_path} must contain one of: target_name, target, label")


def _read_samples(
    data_root: str | Path,
    split_csv: str | Path,
    split: str,
    classes: Iterable[str],
) -> list[VolumeSample]:
    data_root = Path(data_root)
    split_csv = Path(split_csv)
    df = pd.read_csv(split_csv)
    missing = [column for column in ("seriesuid", "image_path", "split") if column not in df.columns]
    if missing:
        raise ValueError(f"{split_csv} is missing required columns: {missing}")

    label_col = _label_column(df, split_csv)
    class_names = [_normalise_class_name(c) for c in classes]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    samples: list[VolumeSample] = []
    df = df[df["split"].astype(str) == split].copy()
    for _, row in df.iterrows():
        class_name = _normalise_class_name(row[label_col])
        if class_name not in class_to_idx:
            continue
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = data_root / image_path
        samples.append(
            VolumeSample(
                seriesuid=str(row["seriesuid"]),
                image_path=image_path,
                label=class_to_idx[class_name],
                class_name=class_name,
                split=split,
                sample_weight=float(row.get("sample_weight", 1.0)),
            )
        )
    if not samples:
        raise RuntimeError(f"No samples found for split '{split}' in {split_csv}")
    return samples


def _window_to_uint8(volume: np.ndarray, center: float = -600.0, width: float = 1500.0) -> np.ndarray:
    low = center - width / 2
    high = center + width / 2
    volume = np.clip(volume.astype(np.float32), low, high)
    volume = (volume - low) / max(high - low, 1e-6) * 255.0
    return np.clip(volume, 0, 255).astype(np.uint8)


class SliceAttentionLuna16Dataset(Dataset):
    """Patient-level dataset returning all axial slices from each preprocessed volume."""

    def __init__(
        self,
        data_root: str | Path,
        split_csv: str | Path,
        split: str,
        image_size: tuple[int, int] = (256, 384),
        classes: Iterable[str] = ("benign", "malignant"),
        train: bool = False,
    ) -> None:
        self.samples = _read_samples(data_root=data_root, split_csv=split_csv, split=split, classes=classes)
        self.transform = build_slice_transform(image_size=image_size, train=train)
        self.labels = [sample.label for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = sitk.ReadImage(str(sample.image_path))
        volume = sitk.GetArrayFromImage(image).astype(np.float32)
        volume_u8 = _window_to_uint8(volume)

        slices = []
        for slice_2d in volume_u8:
            pil = Image.fromarray(slice_2d, mode="L").convert("RGB")
            slices.append(self.transform(pil))
        if not slices:
            raise RuntimeError(f"Empty volume for {sample.seriesuid}: {sample.image_path}")

        volume_tensor = torch.stack(slices, dim=0)
        label = torch.tensor(sample.label, dtype=torch.long)
        return volume_tensor, label, sample.seriesuid

    def get_sampler(self):
        labels = np.array(self.labels)
        neg_count = max(int((labels == 0).sum()), 1)
        pos_count = max(int((labels == 1).sum()), 1)
        class_weight = [1 / neg_count, 1 / pos_count]
        weights = [class_weight[sample.label] * sample.sample_weight for sample in self.samples]
        return WeightedRandomSampler(torch.tensor(weights).double(), len(weights), replacement=True)


def collate_slice_bags(batch):
    max_slices = max(item[0].shape[0] for item in batch)
    channels, height, width = batch[0][0].shape[1:]
    bags = []
    masks = []
    labels = []
    sample_ids = []
    for volume, label, sample_id in batch:
        n_slices = volume.shape[0]
        padded = torch.zeros(max_slices, channels, height, width, dtype=volume.dtype)
        padded[:n_slices] = volume
        mask = torch.zeros(max_slices, dtype=torch.bool)
        mask[:n_slices] = True
        bags.append(padded)
        masks.append(mask)
        labels.append(label)
        sample_ids.append(sample_id)
    return torch.stack(bags), torch.stack(masks), torch.stack(labels), sample_ids
