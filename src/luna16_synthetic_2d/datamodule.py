from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import SyntheticLuna16Dataset


def _as_hw(image_size: int | list[int] | tuple[int, ...]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    if len(image_size) == 1:
        return (int(image_size[0]), int(image_size[0]))
    if len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    raise ValueError(f"image_size must contain one or two integers, got {image_size}")


def build_transforms(image_size: int | list[int] | tuple[int, ...], train: bool):
    resize_size = _as_hw(image_size)
    if train:
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class SyntheticLuna16DataModule(pl.LightningDataModule):
    def __init__(
        self,
        manifest_csv: str,
        split_csv: str | None,
        splits_dir: str,
        fold: int,
        classes: list[str],
        train_split: str,
        val_split: str,
        test_split: str,
        image_size: int | list[int] | tuple[int, ...],
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.manifest_csv = manifest_csv
        self.split_csv = split_csv or f"{splits_dir}/luna16_classification_fold{fold}.csv"
        self.classes = classes
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _build_dataset(self, split: str, train: bool):
        return SyntheticLuna16Dataset(
            manifest_csv=self.manifest_csv,
            split_csv=self.split_csv,
            split=split,
            classes=self.classes,
            transform=build_transforms(self.image_size, train=train),
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset(self.train_split, train=True)
            self.val_dataset = self._build_dataset(self.val_split, train=False)
        if stage in (None, "test"):
            self.test_dataset = self._build_dataset(self.test_split, train=False)

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
