from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import sys

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from dataload.dataset import collate_fn_dict
from dataload.dataset_luna16 import Luna16CPMNetEvalDataset, Luna16CPMNetTrainDataset
from dataload.split_combine import SplitComb


class Luna16CPMNetDataModule(pl.LightningDataModule):
    """CPMNetv2 data module for the preprocessed LUNA16 NIfTI layout."""

    def __init__(
        self,
        images_dir: str,
        train_cases: Sequence[str],
        val_cases: Sequence[str],
        test_cases: Sequence[str],
        labels_csv: str,
        batch_size: int = 1,
        num_workers: int = 0,
        crop_size: Sequence[int] = (96, 96, 96),
        overlap_size: Sequence[int] = (24, 24, 24),
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
        num_samples: int = 1,
        blank_side: int = 0,
        pin_memory: bool = False,
        val_full_volume: bool = False,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.train_cases = list(train_cases)
        self.val_cases = list(val_cases)
        self.test_cases = list(test_cases)
        self.labels_csv = labels_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = list(crop_size)
        self.overlap_size = list(overlap_size)
        self.spacing = list(spacing)
        self.num_samples = num_samples
        self.blank_side = blank_side
        self.pin_memory = pin_memory
        self.val_full_volume = val_full_volume

    @classmethod
    def from_split_csv(cls, csv_path: str, images_dir: str, labels_csv: str | None = None, **kwargs):
        df = pd.read_csv(csv_path)
        if "seriesuid" not in df.columns or "split" not in df.columns:
            raise ValueError(f"{csv_path} must contain seriesuid and split columns.")
        train_cases = df[df["split"] == "train"]["seriesuid"].drop_duplicates().tolist()
        val_cases = df[df["split"] == "val"]["seriesuid"].drop_duplicates().tolist()
        test_cases = df[df["split"] == "test"]["seriesuid"].drop_duplicates().tolist()
        return cls(
            images_dir=images_dir,
            train_cases=train_cases,
            val_cases=val_cases,
            test_cases=test_cases,
            labels_csv=labels_csv or csv_path,
            **kwargs,
        )

    def _split_comb(self):
        return SplitComb(crop_size=self.crop_size, overlap=self.overlap_size, pad_value=-1)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_ds = Luna16CPMNetTrainDataset(
                images_dir=self.images_dir,
                case_list=self.train_cases,
                crop_size=self.crop_size,
                spacing=self.spacing,
                num_samples=self.num_samples,
                csv_file=self.labels_csv,
                blank_side=self.blank_side,
            )
            if self.val_full_volume:
                self.val_ds = Luna16CPMNetEvalDataset(
                    images_dir=self.images_dir,
                    case_list=self.val_cases,
                    splitcomb=self._split_comb(),
                    csv_file=self.labels_csv,
                )
            else:
                self.val_ds = Luna16CPMNetTrainDataset(
                    images_dir=self.images_dir,
                    case_list=self.val_cases,
                    crop_size=self.crop_size,
                    spacing=self.spacing,
                    num_samples=self.num_samples,
                    csv_file=self.labels_csv,
                    blank_side=self.blank_side,
                )
        if stage in (None, "test"):
            self.test_ds = Luna16CPMNetEvalDataset(
                images_dir=self.images_dir,
                case_list=self.test_cases,
                splitcomb=self._split_comb(),
                csv_file=self.labels_csv,
            )

    def _loader_kwargs(self):
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_dict,
            drop_last=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        if self.val_full_volume:
            return DataLoader(self.val_ds, batch_size=1, shuffle=False, **self._loader_kwargs())
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_dict,
            drop_last=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, shuffle=False, **self._loader_kwargs())
