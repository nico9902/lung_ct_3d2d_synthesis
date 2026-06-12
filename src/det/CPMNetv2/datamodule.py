import os
from pathlib import Path
from typing import Optional, Sequence

import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parent
import sys

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import transform
from dataload.crop import InstanceCrop
from dataload.dataset import DetDatasetCSVR, DetDatasetCSVRTest, collate_fn_dict
from dataload.split_combine import SplitComb


class CPMNetv2DataModule(pl.LightningDataModule):
    """Lightning data module for the original CPMNetv2 CSV/NIfTI pipeline."""

    def __init__(
        self,
        root: str,
        train_csv: str = "train_refine.csv",
        val_csv: str = "val.csv",
        test_csv: Optional[str] = None,
        train_images_dir: str = "imagesTr",
        val_images_dir: str = "imagesVa",
        test_images_dir: str = "imagesTs",
        batch_size: int = 1,
        num_workers: int = 0,
        crop_size: Sequence[int] = (64, 128, 128),
        overlap_size: Sequence[int] = (16, 32, 32),
        spacing: Sequence[float] = (0.7, 0.3125, 0.3125),
        num_samples: int = 1,
        blank_side: int = 0,
        pin_memory: bool = False,
        lesion_label: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.root = root
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv or val_csv
        self.train_images_dir = train_images_dir
        self.val_images_dir = val_images_dir
        self.test_images_dir = test_images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = list(crop_size)
        self.overlap_size = list(overlap_size)
        self.spacing = list(spacing)
        self.num_samples = num_samples
        self.blank_side = blank_side
        self.pin_memory = pin_memory
        self.lesion_label = list(lesion_label) if lesion_label else None

    def _csv_path(self, csv_file: str) -> str:
        return csv_file if os.path.isabs(csv_file) else os.path.join(self.root, csv_file)

    def _root(self, image_dir: str) -> str:
        return image_dir if os.path.isabs(image_dir) else os.path.join(self.root, image_dir)

    def _train_transform(self):
        return torchvision.transforms.Compose(
            [
                transform.RandomFlip(flip_depth=True, flip_height=True, flip_width=True, p=0.5),
                transform.RandomTranspose(p=0.5, trans_xy=True, trans_zx=False, trans_zy=False),
                transform.Pad(output_size=self.crop_size),
                transform.RandomCrop(output_size=self.crop_size, pos_ratio=0.9),
                transform.CoordToAnnot(blank_side=self.blank_side),
            ]
        )

    def _train_crop(self):
        return InstanceCrop(
            crop_size=self.crop_size,
            tp_ratio=0.75,
            spacing=self.spacing,
            rand_trans=[10, 20, 20],
            rand_rot=[20, 0, 0],
            rand_space=[0.9, 1.2],
            sample_num=self.num_samples,
            blank_side=self.blank_side,
            instance_crop=True,
        )

    def _split_comb(self):
        return SplitComb(crop_size=self.crop_size, overlap=self.overlap_size, pad_value=-1)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_ds = DetDatasetCSVR(
                roots=[self._root(self.train_images_dir)],
                crop_fn=self._train_crop(),
                transform_post=self._train_transform(),
                csv_file=self._csv_path(self.train_csv),
                lesion_label=self.lesion_label,
            )
            self.val_ds = DetDatasetCSVRTest(
                roots=[self._root(self.val_images_dir)],
                SplitComb=self._split_comb(),
                csv_file=self._csv_path(self.val_csv),
                lesion_label=self.lesion_label,
            )

        if stage in (None, "test"):
            self.test_ds = DetDatasetCSVRTest(
                roots=[self._root(self.test_images_dir)],
                SplitComb=self._split_comb(),
                csv_file=self._csv_path(self.test_csv),
                lesion_label=self.lesion_label,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_dict,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

