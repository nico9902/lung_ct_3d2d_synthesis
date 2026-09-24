"""LUNA16 patient-level sliding-window DataModule for the 3DINO baseline.

Reuses ``Luna16VolumeDataset`` from the existing 3D ResNet18 full-volume
baseline (``src/luna16_volume_3d/train_resnet18.py``) for CT loading -- same
fold CSVs, same label/patient filtering, same preprocessed-volume convention
(1mm-isotropic, lung-windowed, ``[0, 255] -> [0, 1]``) already used elsewhere
in this repo -- rather than a new parallel loader. Each patient's native
(un-resized) volume is split into overlapping 3D windows at 3DINO's native
112^3 input size; windows are the *only* representation, with no
lesion-guided crop/mask/detection input anywhere in the pipeline.

One "batch" is one patient (all of that patient's windows), so window
aggregation never mixes patients; ``accumulate_grad_batches`` in the
training script is used to reach a larger effective batch size across
patients.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.luna16_3dino_3d.model import NATIVE_INPUT_SIZE, extract_windows, percentile_normalize  # noqa: E402
from src.luna16_volume_3d.train_resnet18 import Luna16VolumeDataset  # noqa: E402


class Luna16WindowDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split_csv: str | Path,
        split: str,
        window_size: int = NATIVE_INPUT_SIZE,
        stride: int = NATIVE_INPUT_SIZE // 2,
        max_windows: Optional[int] = None,
        windows_cache_dir: str | Path | None = None,
    ) -> None:
        self.volumes = Luna16VolumeDataset(
            data_root=data_root,
            split_csv=split_csv,
            split=split,
            volume_size=None,  # native shape: windowing happens below, no resize
            train=False,  # no flips/noise: sliding windows already vary spatial content
        )
        self.window_size = window_size
        self.stride = stride
        self.max_windows = max_windows
        self.labels = self.volumes.labels

        # Windows depend only on (patient, window_size, stride), not on fold or
        # epoch: cache the preprocessed (post-normalization) window tensor per
        # patient so repeated CT loading/windowing/percentile-normalization
        # (measured as the dominant per-patient cost, far above raw GPU
        # compute) happens once across all 10 folds and all training epochs.
        self.windows_cache_dir = Path(windows_cache_dir) if windows_cache_dir else None
        if self.windows_cache_dir is not None:
            self.windows_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.volumes)

    def _cache_path(self, seriesuid: str) -> Path:
        return self.windows_cache_dir / f"{seriesuid}_w{self.window_size}_s{self.stride}.pt"

    def __getitem__(self, index: int):
        sample_id = str(self.volumes.df.iloc[index]["seriesuid"])
        label = torch.tensor(self.labels[index], dtype=torch.long)

        cache_path = self._cache_path(sample_id) if self.windows_cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            windows = torch.load(cache_path, map_location="cpu").float()
        else:
            tensor, _label, _sample_id = self.volumes[index]  # tensor: (1, D, H, W) in [0, 1]
            volume = tensor.squeeze(0)
            windows, _starts = extract_windows(volume, self.window_size, self.stride)
            windows = percentile_normalize(windows)
            if cache_path is not None:
                tmp_path = cache_path.with_suffix(f".tmp{os.getpid()}.pt")
                torch.save(windows.half(), tmp_path)
                tmp_path.replace(cache_path)

        if self.max_windows is not None and windows.shape[0] > self.max_windows:
            keep = torch.linspace(0, windows.shape[0] - 1, self.max_windows).round().long()
            windows = windows[keep]
        return windows, label, sample_id

    def sampler(self):
        return self.volumes.sampler()


def _single_patient_collate(batch):
    if len(batch) != 1:
        raise ValueError("Luna16WindowDataset must be used with batch_size=1 (one patient per step).")
    return batch[0]


class Luna163DinoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        splits_dir: str,
        fold: int,
        window_size: int = NATIVE_INPUT_SIZE,
        stride: int = NATIVE_INPUT_SIZE // 2,
        max_windows: Optional[int] = None,
        windows_cache_dir: str | None = None,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.split_csv = Path(splits_dir) / f"luna16_classification_fold{fold}.csv"

    def setup(self, stage: Optional[str] = None) -> None:
        common = dict(
            data_root=self.hparams.data_root,
            split_csv=self.split_csv,
            window_size=self.hparams.window_size,
            stride=self.hparams.stride,
            max_windows=self.hparams.max_windows,
            windows_cache_dir=self.hparams.windows_cache_dir,
        )
        if stage in (None, "fit"):
            self.train_dataset = Luna16WindowDataset(split="train", **common)
            self.val_dataset = Luna16WindowDataset(split="val", **common)
        if stage in (None, "fit", "test"):
            self.test_dataset = Luna16WindowDataset(split="test", **common)

    def _loader(self, dataset: Luna16WindowDataset, shuffle: bool, sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            prefetch_factor=2 if self.hparams.num_workers > 0 else None,
            collate_fn=_single_patient_collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=False, sampler=self.train_dataset.sampler())

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
