from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .dataset import Luna16SliceDataset, collate_slice_stacks


def _as_hw(image_size: int | list[int] | tuple[int, ...]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    if len(image_size) == 1:
        return (int(image_size[0]), int(image_size[0]))
    if len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    raise ValueError(f"image_size must contain one or two integers, got {image_size}")


class Luna16SliceAttentionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        splits_dir: str,
        fold: int,
        image_size: int | list[int] | tuple[int, ...],
        batch_size: int,
        num_workers: int,
        cache_dir: str | None = None,
        cache_read_only: bool = False,
        limit_train_samples: int | None = None,
        limit_val_samples: int | None = None,
        limit_test_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.split_csv = Path(splits_dir) / f"luna16_classification_fold{fold}.csv"
        self.image_size = _as_hw(image_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.cache_read_only = cache_read_only
        self.cache_write_enabled = mp.Value("b", not cache_read_only)
        self.limit_train_samples = limit_train_samples
        self.limit_val_samples = limit_val_samples
        self.limit_test_samples = limit_test_samples

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = Luna16SliceDataset(
            self.data_root,
            self.split_csv,
            "train",
            self.image_size,
            train=True,
            cache_dir=self.cache_dir,
            cache_read_only=self.cache_read_only,
            cache_write_enabled=self.cache_write_enabled,
            limit_samples=self.limit_train_samples,
        )
        self.val_dataset = Luna16SliceDataset(
            self.data_root,
            self.split_csv,
            "val",
            self.image_size,
            train=False,
            cache_dir=self.cache_dir,
            cache_read_only=self.cache_read_only,
            cache_write_enabled=self.cache_write_enabled,
            limit_samples=self.limit_val_samples,
        )
        self.test_dataset = Luna16SliceDataset(
            self.data_root,
            self.split_csv,
            "test",
            self.image_size,
            train=False,
            cache_dir=self.cache_dir,
            cache_read_only=self.cache_read_only,
            cache_write_enabled=self.cache_write_enabled,
            limit_samples=self.limit_test_samples,
        )

    def _loader(self, dataset, shuffle: bool = False, sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=1 if self.num_workers > 0 else None,
            collate_fn=collate_slice_stacks,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, sampler=self.train_dataset.sampler())

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)

    def disable_cache_writes(self) -> None:
        with self.cache_write_enabled.get_lock():
            self.cache_write_enabled.value = False

    def complete_missing_compact_cache(self) -> tuple[int, int]:
        """Create only missing uint8 cache entries across train, validation, and test."""
        if self.cache_dir is None:
            raise RuntimeError("--cache-write-first-epoch-only requires --cache-dir.")
        if self.cache_read_only:
            raise RuntimeError("Cannot complete the cache while --cache-read-only is enabled.")

        tasks = []
        seen: set[str] = set()
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            for index, seriesuid in enumerate(dataset.df["seriesuid"].astype(str)):
                if seriesuid in seen:
                    continue
                seen.add(seriesuid)
                cache_path = dataset.compact_cache_path(seriesuid)
                if cache_path is not None and not cache_path.exists():
                    tasks.append((dataset, index))

        print(
            f"Compact cache pre-pass: {len(seen)} expected, {len(tasks)} missing.",
            flush=True,
        )
        if not tasks:
            return len(seen), 0

        def create_entry(task) -> bool:
            dataset, index = task
            return dataset.ensure_compact_cache(index)

        workers = max(1, self.num_workers)
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for created in executor.map(create_entry, tasks):
                completed += int(created)
                if completed % 25 == 0 or completed == len(tasks):
                    print(
                        f"Compact cache pre-pass: created {completed}/{len(tasks)} missing entries.",
                        flush=True,
                    )

        remaining = 0
        for dataset, index in tasks:
            seriesuid = str(dataset.df.iloc[index]["seriesuid"])
            cache_path = dataset.compact_cache_path(seriesuid)
            remaining += int(cache_path is None or not cache_path.exists())
        if remaining:
            raise RuntimeError(f"Compact cache pre-pass left {remaining} missing entries.")
        return len(seen), completed
