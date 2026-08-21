from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .dataset import _normalise_class_name


def feature_filename(seriesuid: str) -> str:
    safe = str(seriesuid).replace("/", "_").replace("\\", "_")
    return f"{safe}.pt"


def _label_column(df: pd.DataFrame, csv_path: Path) -> str:
    for column in ("target_name", "target", "label"):
        if column in df.columns:
            return column
    raise ValueError(f"{csv_path} must contain one of: target_name, target, label")


class SliceFeatureDataset(Dataset):
    def __init__(
        self,
        feature_dir: str | Path,
        split_csv: str | Path,
        split: str,
        classes: Iterable[str] = ("benign", "malignant"),
    ) -> None:
        self.feature_dir = Path(feature_dir)
        self.split_csv = Path(split_csv)
        df = pd.read_csv(self.split_csv)
        label_col = _label_column(df, self.split_csv)
        class_names = [_normalise_class_name(c) for c in classes]
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        rows = df[df["split"].astype(str) == split].copy()

        self.samples = []
        for _, row in rows.iterrows():
            class_name = _normalise_class_name(row[label_col])
            if class_name not in class_to_idx:
                continue
            seriesuid = str(row["seriesuid"])
            feature_path = self.feature_dir / feature_filename(seriesuid)
            if not feature_path.exists():
                raise FileNotFoundError(f"Missing cached features for {seriesuid}: {feature_path}")
            self.samples.append(
                {
                    "seriesuid": seriesuid,
                    "feature_path": feature_path,
                    "label": class_to_idx[class_name],
                    "class_name": class_name,
                    "sample_weight": float(row.get("sample_weight", 1.0)),
                }
            )
        if not self.samples:
            raise RuntimeError(f"No samples found for split '{split}' in {self.split_csv}")
        self.labels = [sample["label"] for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        payload = torch.load(sample["feature_path"], map_location="cpu")
        features = payload["features"].float()
        label = torch.tensor(sample["label"], dtype=torch.long)
        return features, label, sample["seriesuid"]

    def get_sampler(self):
        labels = np.array(self.labels)
        neg_count = max(int((labels == 0).sum()), 1)
        pos_count = max(int((labels == 1).sum()), 1)
        class_weight = [1 / neg_count, 1 / pos_count]
        weights = [class_weight[sample["label"]] * sample["sample_weight"] for sample in self.samples]
        return WeightedRandomSampler(torch.tensor(weights).double(), len(weights), replacement=True)


def collate_feature_bags(batch):
    max_slices = max(item[0].shape[0] for item in batch)
    feature_dim = batch[0][0].shape[1]
    bags = []
    masks = []
    labels = []
    sample_ids = []
    for features, label, sample_id in batch:
        n_slices = features.shape[0]
        padded = torch.zeros(max_slices, feature_dim, dtype=features.dtype)
        padded[:n_slices] = features
        mask = torch.zeros(max_slices, dtype=torch.bool)
        mask[:n_slices] = True
        bags.append(padded)
        masks.append(mask)
        labels.append(label)
        sample_ids.append(sample_id)
    return torch.stack(bags), torch.stack(masks), torch.stack(labels), sample_ids
