from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .dataset import DetectionMILDataset


class DetectionMILDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        split_csv: str | None,
        splits_dir: str,
        predictions_root: str,
        prediction_name: str,
        fold: int,
        classes: list[str],
        train_split: str,
        val_split: str,
        test_split: str,
        top_k: int,
        min_probability: float,
        crop_size_mm: int,
        crop_image_size: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.split_csv = split_csv or f"{splits_dir}/luna16_classification_fold{fold}.csv"
        self.predictions_root = predictions_root
        self.prediction_name = prediction_name
        self.fold = fold
        self.classes = classes
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.top_k = top_k
        self.min_probability = min_probability
        self.crop_size_mm = crop_size_mm
        self.crop_image_size = crop_image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_counts = None

    def _build_dataset(self, split: str, train: bool):
        return DetectionMILDataset(
            data_root=self.data_root,
            split_csv=self.split_csv,
            predictions_root=self.predictions_root,
            fold=self.fold,
            split=split,
            top_k=self.top_k,
            min_probability=self.min_probability,
            crop_size_mm=self.crop_size_mm,
            crop_image_size=self.crop_image_size,
            classes=self.classes,
            prediction_name=self.prediction_name,
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
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=False, sampler=self.train_dataset.get_sampler())

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
