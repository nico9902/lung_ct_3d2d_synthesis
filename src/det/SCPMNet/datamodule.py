from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.det.SCPMNet.dataset import SCPMCSVVolumeDataset, SCPMSlidingWindowDataset, scpm_collate, scpm_sliding_collate


def _require_non_empty(dataset, split: str, csv_path: str, data_root: str) -> None:
    if len(dataset) > 0:
        return
    raise ValueError(
        f"SCPM-Net {split} dataset is empty. "
        f"Check csv_path={csv_path!r}, data_root={data_root!r}, split={split!r}, "
        "and image_path/seriesuid volume locations. If paths are intentionally remote or mounted later, "
        "set skip_missing_images=false."
    )


class SCPMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        data_root: str = "",
        batch_size: int = 1,
        num_workers: int = 0,
        crop_size: tuple[int, int, int] = (96, 96, 96),
        samples_per_volume: int = 1,
        clip: tuple[float, float] = (-1000.0, 400.0),
        intensity_mode: str = "hu",
        normalized_volume_cache_dir: str | None = None,
        positive_crop_prob: float = 0.7,
        mask_path_column: str | None = None,
        skip_missing_images: bool = True,
        val_full_volume: bool = False,
        val_modes: tuple[str, ...] | list[str] | None = None,
        val_fixed_crop_seed: int | None = None,
        val_random_crop_samples_per_volume: int = 1,
        test_full_volume: bool = True,
        sliding_window_stride: tuple[int, int, int] = (24, 24, 24),
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        kwargs = dict(
            csv_path=self.hparams.csv_path,
            data_root=self.hparams.data_root,
            crop_size=tuple(self.hparams.crop_size),
            samples_per_volume=self.hparams.samples_per_volume,
            clip=tuple(self.hparams.clip),
            intensity_mode=self.hparams.intensity_mode,
            normalized_volume_cache_dir=self.hparams.normalized_volume_cache_dir,
            positive_crop_prob=self.hparams.positive_crop_prob,
            mask_path_column=self.hparams.mask_path_column,
            skip_missing_images=self.hparams.skip_missing_images,
        )
        if stage in (None, "fit", "validate"):
            self.train_ds = SCPMCSVVolumeDataset(split="train", **kwargs)
            val_modes = [str(mode) for mode in (self.hparams.val_modes or [])]
            self.val_loader_names = []
            self.val_collate_fns = []
            if val_modes:
                self.val_ds = []
                for mode in val_modes:
                    if mode == "full_volume_loss":
                        dataset = SCPMSlidingWindowDataset(
                            csv_path=self.hparams.csv_path,
                            split="val",
                            data_root=self.hparams.data_root,
                            crop_size=tuple(self.hparams.crop_size),
                            stride=tuple(self.hparams.sliding_window_stride),
                            clip=tuple(self.hparams.clip),
                            intensity_mode=self.hparams.intensity_mode,
                            normalized_volume_cache_dir=self.hparams.normalized_volume_cache_dir,
                            mask_path_column=self.hparams.mask_path_column,
                            skip_missing_images=self.hparams.skip_missing_images,
                            include_annotations=True,
                        )
                        name = "full_volume"
                        collate_fn = scpm_collate
                    elif mode == "fixed_crop_loss":
                        fixed_kwargs = dict(kwargs)
                        fixed_kwargs["samples_per_volume"] = 1
                        dataset = SCPMCSVVolumeDataset(
                            split="val",
                            deterministic_seed=self.hparams.val_fixed_crop_seed,
                            **fixed_kwargs,
                        )
                        name = "fixed_crop"
                        collate_fn = scpm_collate
                    elif mode == "random_crop_loss":
                        random_kwargs = dict(kwargs)
                        random_kwargs["samples_per_volume"] = self.hparams.val_random_crop_samples_per_volume
                        dataset = SCPMCSVVolumeDataset(split="val", **random_kwargs)
                        name = "random_crop"
                        collate_fn = scpm_collate
                    elif mode == "full_volume_froc":
                        dataset = SCPMSlidingWindowDataset(
                            csv_path=self.hparams.csv_path,
                            split="val",
                            data_root=self.hparams.data_root,
                            crop_size=tuple(self.hparams.crop_size),
                            stride=tuple(self.hparams.sliding_window_stride),
                            clip=tuple(self.hparams.clip),
                            intensity_mode=self.hparams.intensity_mode,
                            normalized_volume_cache_dir=self.hparams.normalized_volume_cache_dir,
                            include_annotations=True,
                            skip_missing_images=self.hparams.skip_missing_images,
                        )
                        name = "full_volume_froc"
                        collate_fn = scpm_sliding_collate
                    else:
                        raise ValueError(f"Unsupported SCPM-Net validation mode: {mode}")
                    self.val_ds.append(dataset)
                    self.val_loader_names.append(name)
                    self.val_collate_fns.append(collate_fn)
                for dataset, name in zip(self.val_ds, self.val_loader_names):
                    _require_non_empty(dataset, f"val:{name}", self.hparams.csv_path, self.hparams.data_root)
            elif self.hparams.val_full_volume:
                self.val_ds = SCPMSlidingWindowDataset(
                    csv_path=self.hparams.csv_path,
                    split="val",
                    data_root=self.hparams.data_root,
                    crop_size=tuple(self.hparams.crop_size),
                    stride=tuple(self.hparams.sliding_window_stride),
                    clip=tuple(self.hparams.clip),
                    intensity_mode=self.hparams.intensity_mode,
                    normalized_volume_cache_dir=self.hparams.normalized_volume_cache_dir,
                    skip_missing_images=self.hparams.skip_missing_images,
                )
                self.val_loader_names = ["val"]
                self.val_collate_fns = [scpm_sliding_collate]
            else:
                self.val_ds = SCPMCSVVolumeDataset(split="val", **kwargs)
                self.val_loader_names = ["val"]
                self.val_collate_fns = [scpm_collate]
            _require_non_empty(self.train_ds, "train", self.hparams.csv_path, self.hparams.data_root)
            if not isinstance(self.val_ds, list):
                _require_non_empty(self.val_ds, "val", self.hparams.csv_path, self.hparams.data_root)
        if stage in (None, "test", "predict"):
            if self.hparams.test_full_volume:
                self.test_ds = SCPMSlidingWindowDataset(
                    csv_path=self.hparams.csv_path,
                    split="test",
                    data_root=self.hparams.data_root,
                    crop_size=tuple(self.hparams.crop_size),
                    stride=tuple(self.hparams.sliding_window_stride),
                    clip=tuple(self.hparams.clip),
                    intensity_mode=self.hparams.intensity_mode,
                    normalized_volume_cache_dir=self.hparams.normalized_volume_cache_dir,
                    skip_missing_images=self.hparams.skip_missing_images,
                )
            else:
                self.test_ds = SCPMCSVVolumeDataset(split="test", **kwargs)

    def _loader_kwargs(self) -> dict:
        kwargs = {"num_workers": self.hparams.num_workers, "pin_memory": self.hparams.pin_memory}
        if self.hparams.num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=scpm_collate,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.val_ds, list):
            return [
                DataLoader(
                    dataset,
                    batch_size=self.hparams.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    **self._loader_kwargs(),
                )
                for dataset, collate_fn in zip(self.val_ds, self.val_collate_fns)
            ]
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.val_collate_fns[0],
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        collate_fn = scpm_sliding_collate if self.hparams.test_full_volume else scpm_collate
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **self._loader_kwargs(),
        )
