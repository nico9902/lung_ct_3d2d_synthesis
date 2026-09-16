from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler


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


class Luna16SliceDataset(Dataset):
    """Return a preprocessed LUNA16 volume as a stack of axial 2D slices."""

    def __init__(
        self,
        data_root: str | Path,
        split_csv: str | Path,
        split: str,
        image_size: tuple[int, int],
        train: bool,
        limit_samples: int | None = None,
        cache_dir: str | Path | None = None,
        cache_read_only: bool = False,
        cache_write_enabled=None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split_csv = Path(split_csv)
        self.split = split
        self.image_size = image_size
        self.train = train
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_read_only = cache_read_only
        self.cache_write_enabled = cache_write_enabled
        if self.cache_dir is not None and not self.cache_read_only:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.split_csv)
        df = df[(df["split"].astype(str) == split) & (df["target"].isin([0, 1]))].copy()
        df["target"] = df["target"].astype(int)
        if limit_samples is not None:
            df = df.head(limit_samples)
        if df.empty:
            raise RuntimeError(f"No binary samples found for split={split} in {self.split_csv}")
        self.df = df.reset_index(drop=True)
        self.labels = self.df["target"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        slices = self._load_slices(row)
        label = torch.tensor(int(row["target"]), dtype=torch.long)
        return slices, label, str(row["seriesuid"])

    def _save_cache(self, tensor: torch.Tensor, cache_path: Path) -> None:
        writes_enabled = self.cache_write_enabled is None or bool(self.cache_write_enabled.value)
        if self.cache_read_only or not writes_enabled:
            return
        tmp_path = cache_path.with_name(f"{cache_path.name}.{random.getrandbits(32):08x}.tmp")
        try:
            torch.save(tensor, tmp_path)
            if cache_path.exists():
                tmp_path.unlink()
            else:
                tmp_path.replace(cache_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _as_uint8(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == torch.uint8:
            return tensor
        return (
            tensor.float()
            .nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
            .clamp_(0.0, 1.0)
            .mul_(255.0)
            .round_()
            .to(torch.uint8)
        )

    def compact_cache_path(self, seriesuid: str) -> Path | None:
        if self.cache_dir is None:
            return None
        height, width = self.image_size
        return self.cache_dir / f"{seriesuid}_axial_{height}x{width}_uint8.pt"

    def ensure_compact_cache(self, index: int) -> bool:
        """Create one missing uint8 entry; return True only when a file was created."""
        row = self.df.iloc[index]
        cache_path = self.compact_cache_path(str(row["seriesuid"]))
        if cache_path is None:
            raise RuntimeError("Cannot complete the compact cache without a cache directory.")
        if cache_path.exists():
            return False
        self._load_slices(row)
        if not cache_path.exists():
            raise RuntimeError(f"Compact cache entry was not written: {cache_path}")
        return True

    def _load_slices(self, row: pd.Series) -> torch.Tensor:
        cache_path = None
        legacy_cache_path = None
        if self.cache_dir is not None:
            height, width = self.image_size
            stem = f"{row['seriesuid']}_axial_{height}x{width}"
            cache_path = self.compact_cache_path(str(row["seriesuid"]))
            legacy_cache_path = self.cache_dir / f"{stem}.pt"
            if cache_path.exists():
                return self._as_uint8(torch.load(cache_path, map_location="cpu"))
            if legacy_cache_path.exists():
                # Migrate from the existing normalized FP16/FP32 cache without
                # reading or preprocessing the source medical image.
                tensor = self._as_uint8(torch.load(legacy_cache_path, map_location="cpu"))
                self._save_cache(tensor, cache_path)
                return tensor

        import SimpleITK as sitk

        image_path = self.data_root / row["image_path"]
        image = sitk.ReadImage(str(image_path))
        volume = sitk.GetArrayFromImage(image).astype(np.float32)
        volume = np.nan_to_num(volume, nan=0.0, posinf=255.0, neginf=0.0)
        volume = np.clip(volume, 0.0, 255.0) / 255.0

        tensor = torch.from_numpy(volume).unsqueeze(1)
        tensor = F.interpolate(
            tensor,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor.contiguous()

        tensor = self._as_uint8(tensor)
        if cache_path is not None:
            self._save_cache(tensor, cache_path)
        return tensor

    def sampler(self) -> WeightedRandomSampler:
        labels = torch.tensor(self.labels, dtype=torch.long)
        counts = torch.bincount(labels, minlength=2).float().clamp_min(1.0)
        weights = 1.0 / counts[labels]
        return WeightedRandomSampler(weights.double(), num_samples=len(weights), replacement=True)


def collate_slice_stacks(batch):
    """Pack variable-depth patients without padding their image tensors."""
    slices, labels, sample_ids = zip(*batch)
    slice_counts = torch.tensor([stack.shape[0] for stack in slices], dtype=torch.long)
    return torch.cat(slices, dim=0), slice_counts, torch.stack(labels), list(sample_ids)
