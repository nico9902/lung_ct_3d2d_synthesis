from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class DetectionCandidate:
    z: float
    y: float
    x: float
    radius: float
    probability: float


@dataclass(frozen=True)
class PatientSample:
    seriesuid: str
    image_path: Path
    label: int
    class_name: str
    split: str
    candidates: tuple[DetectionCandidate, ...]
    sample_weight: float = 1.0


class MinMaxScale:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        min_value = tensor.amin()
        max_value = tensor.amax()
        return (tensor - min_value) / (max_value - min_value).clamp_min(1e-6)


def build_crop_transform(crop_image_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((crop_image_size, crop_image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                MinMaxScale(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((crop_image_size, crop_image_size)),
            transforms.ToTensor(),
            MinMaxScale(),
        ]
    )


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


def _resolve_fold_predictions(predictions_root: Path, fold: int, prediction_name: str) -> Path:
    candidates = [
        predictions_root / f"fold_{fold}" / prediction_name,
        predictions_root / f"fold_{fold}" / "predictions" / prediction_name,
    ]
    candidates.extend(sorted(predictions_root.glob(f"*fold{fold}/predictions/{prediction_name}")))
    candidates.extend(sorted(predictions_root.glob(f"*fold_{fold}/predictions/{prediction_name}")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find predictions for fold {fold} under {predictions_root} "
        f"with prediction file {prediction_name}"
    )


def _load_candidates(
    predictions_root: Path,
    fold: int,
    prediction_name: str,
    top_k: int,
    min_probability: float,
) -> dict[str, tuple[DetectionCandidate, ...]]:
    pred_path = _resolve_fold_predictions(predictions_root, fold, prediction_name)
    df = pd.read_csv(pred_path)
    required = {"seriesuid", "coordZ", "coordY", "coordX", "radius", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{pred_path} is missing required columns: {sorted(missing)}")

    df = df[df["probability"].astype(float) >= float(min_probability)].copy()
    df = df.sort_values(["seriesuid", "probability"], ascending=[True, False])
    by_patient: dict[str, tuple[DetectionCandidate, ...]] = {}
    for seriesuid, group in df.groupby("seriesuid"):
        rows = group.head(top_k)
        by_patient[str(seriesuid)] = tuple(
            DetectionCandidate(
                z=float(row.coordZ),
                y=float(row.coordY),
                x=float(row.coordX),
                radius=float(row.radius),
                probability=float(row.probability),
            )
            for row in rows.itertuples(index=False)
        )
    return by_patient


def _read_samples(
    data_root: str | Path,
    split_csv: str | Path,
    predictions_root: str | Path,
    fold: int,
    prediction_name: str,
    split: str,
    classes: Iterable[str],
    top_k: int,
    min_probability: float,
) -> list[PatientSample]:
    data_root = Path(data_root)
    split_csv = Path(split_csv)
    predictions_root = Path(predictions_root)
    split_df = pd.read_csv(split_csv)
    missing = [column for column in ("seriesuid", "image_path", "split") if column not in split_df.columns]
    if missing:
        raise ValueError(f"{split_csv} is missing required columns: {missing}")

    label_col = _label_column(split_df, split_csv)
    class_names = [_normalise_class_name(c) for c in classes]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    candidates_by_patient = _load_candidates(
        predictions_root=predictions_root,
        fold=fold,
        prediction_name=prediction_name,
        top_k=top_k,
        min_probability=min_probability,
    )

    samples: list[PatientSample] = []
    split_df = split_df[split_df["split"].astype(str) == split].copy()
    for _, row in split_df.iterrows():
        class_name = _normalise_class_name(row[label_col])
        if class_name not in class_to_idx:
            continue
        seriesuid = str(row["seriesuid"])
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = data_root / image_path
        samples.append(
            PatientSample(
                seriesuid=seriesuid,
                image_path=image_path,
                label=class_to_idx[class_name],
                class_name=class_name,
                split=split,
                candidates=candidates_by_patient.get(seriesuid, tuple()),
                sample_weight=float(row.get("sample_weight", 1.0)),
            )
        )
    if not samples:
        raise RuntimeError(f"No samples found for split '{split}' in {split_csv}")
    return samples


def _window_to_uint8(image: np.ndarray, center: float = -600.0, width: float = 1500.0) -> np.ndarray:
    low = center - width / 2
    high = center + width / 2
    image = np.clip(image.astype(np.float32), low, high)
    image = (image - low) / max(high - low, 1e-6) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _crop_with_padding(slice_image: np.ndarray, center_y: float, center_x: float, crop_size: int) -> np.ndarray:
    half = crop_size // 2
    cy = int(round(center_y))
    cx = int(round(center_x))
    y0 = cy - half
    x0 = cx - half
    y1 = y0 + crop_size
    x1 = x0 + crop_size

    out = np.full((crop_size, crop_size), fill_value=-1000.0, dtype=np.float32)
    src_y0 = max(y0, 0)
    src_x0 = max(x0, 0)
    src_y1 = min(y1, slice_image.shape[0])
    src_x1 = min(x1, slice_image.shape[1])
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = slice_image[
            src_y0:src_y1, src_x0:src_x1
        ]
    return out


class DetectionMILDataset(Dataset):
    """Patient-level MIL dataset from detector-centered 2D crops."""

    def __init__(
        self,
        data_root: str | Path,
        split_csv: str | Path,
        predictions_root: str | Path,
        fold: int,
        split: str,
        top_k: int = 4,
        min_probability: float = 0.5,
        crop_size_mm: int = 64,
        crop_image_size: int = 224,
        classes: Iterable[str] = ("benign", "malignant"),
        prediction_name: str = "test_predictions.csv",
        train: bool = False,
    ) -> None:
        self.samples = _read_samples(
            data_root=data_root,
            split_csv=split_csv,
            predictions_root=predictions_root,
            fold=fold,
            prediction_name=prediction_name,
            split=split,
            classes=classes,
            top_k=top_k,
            min_probability=min_probability,
        )
        self.top_k = int(top_k)
        self.crop_size_mm = int(crop_size_mm)
        self.crop_image_size = int(crop_image_size)
        self.transform = build_crop_transform(crop_image_size, train=train)
        self.labels = [sample.label for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = sitk.ReadImage(str(sample.image_path))
        volume = sitk.GetArrayFromImage(image).astype(np.float32)

        crops: list[torch.Tensor] = []
        candidate_probs: list[float] = []
        candidates = list(sample.candidates[: self.top_k])
        for candidate in candidates:
            z = int(np.clip(round(candidate.z), 0, volume.shape[0] - 1))
            crop = _crop_with_padding(volume[z], candidate.y, candidate.x, self.crop_size_mm)
            pil = Image.fromarray(_window_to_uint8(crop), mode="L").convert("RGB")
            crops.append(self.transform(pil))
            candidate_probs.append(float(candidate.probability))

        valid_count = len(crops)
        while len(crops) < self.top_k:
            crops.append(torch.zeros(3, self.crop_image_size, self.crop_image_size))
            candidate_probs.append(0.0)

        bag = torch.stack(crops, dim=0)
        valid_mask = torch.zeros(self.top_k, dtype=torch.bool)
        valid_mask[:valid_count] = True
        label = torch.tensor(sample.label, dtype=torch.long)
        candidate_probs_tensor = torch.tensor(candidate_probs, dtype=torch.float32)
        return bag, valid_mask, candidate_probs_tensor, label, sample.seriesuid

    def get_sampler(self):
        labels = np.array(self.labels)
        neg_class_count = max(int((labels == 0).sum()), 1)
        pos_class_count = max(int((labels == 1).sum()), 1)
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[sample.label] * sample.sample_weight for sample in self.samples]
        return torch.utils.data.WeightedRandomSampler(torch.tensor(weights).double(), len(weights), replacement=True)
