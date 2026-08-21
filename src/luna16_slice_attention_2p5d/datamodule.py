from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .dataset import SliceAttentionLuna16Dataset, collate_slice_bags


class SliceAttentionLuna16DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        split_csv: str | None,
        splits_dir: str,
        fold: int,
        classes: list[str],
        train_split: str,
        val_split: str,
        test_split: str,
        image_size: tuple[int, int],
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.split_csv = split_csv or f"{splits_dir}/luna16_classification_fold{fold}.csv"
        self.fold = fold
        self.classes = classes
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_counts = None

    def _build_dataset(self, split: str, train: bool):
        return SliceAttentionLuna16Dataset(
            data_root=self.data_root,
            split_csv=self.split_csv,
            split=split,
            image_size=self.image_size,
            classes=self.classes,
            train=train,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset(self.train_split, train=True)
            self.val_dataset = self._build_dataset(self.val_split, train=False)
            labels = torch.tensor(self.train_dataset.labels, dtype=torch.long)
            self.class_counts = torch.bincount(labels, minlength=len(self.classes)).long()
        if stage in (None, "test"):
            self.test_dataset = self._build_dataset(self.test_split, train=False)

    def _loader(self, dataset, shuffle: bool, sampler=None):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_slice_bags,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=False, sampler=self.train_dataset.get_sampler())

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
