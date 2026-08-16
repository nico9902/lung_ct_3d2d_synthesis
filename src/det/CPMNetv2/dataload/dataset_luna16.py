from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import Dataset

PACKAGE_ROOT = Path(__file__).resolve().parent
CPMNET_ROOT = PACKAGE_ROOT.parent

for path in (PACKAGE_ROOT, CPMNET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import transform
from dataload.crop import InstanceCrop
from dataload.split_combine import SplitComb


LABEL_COLUMNS = ["seriesuid", "x", "y", "z", "w", "h", "d", "label"]
DEFAULT_LESION_LABELS = ["nodule"]


def _normalize_luna_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype("float32", copy=False)
    max_value = float(np.nanmax(volume)) if volume.size else 0.0
    min_value = float(np.nanmin(volume)) if volume.size else 0.0
    if max_value > 1.5 or min_value < -0.5:
        volume = np.clip(volume, 0.0, 255.0) / 255.0
    return volume


def _load_luna_volume(path: str):
    nii = nib.load(path)
    volume = np.asanyarray(nii.dataobj).astype("float32")
    zoom_x, zoom_y, zoom_z = nii.header.get_zooms()[:3]
    # NIfTI is read as (x, y, z). CPMNetv2 uses (z, y, x).
    volume = volume.transpose(2, 1, 0)
    spacing = np.array([zoom_z, zoom_y, zoom_x], dtype="float32")
    return _normalize_luna_volume(volume), spacing


def _load_labels_csv(csv_file: str):
    if not csv_file:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"LUNA16 label CSV not found: {csv_file}")

    labels = pd.read_csv(csv_file)
    missing = [column for column in LABEL_COLUMNS if column not in labels.columns]
    if missing:
        raise ValueError(f"LUNA16 label CSV is missing columns: {missing}")
    labels = labels[LABEL_COLUMNS]
    labels = labels[labels["label"].astype(str).str.lower().isin(DEFAULT_LESION_LABELS)]
    labels = labels.dropna(subset=["x", "y", "z", "w", "h", "d"])
    return labels


def _labels_to_arrays(csv_label: pd.DataFrame):
    all_loc = csv_label[["z", "y", "x"]].to_numpy(dtype="float32")
    all_rad = csv_label[["d", "h", "w"]].to_numpy(dtype="float32")
    return all_loc, all_rad


class Luna16CPMNetTrainDataset(Dataset):
    """LUNA16 NIfTI adapter for CPMNetv2's original crop-based training path."""

    def __init__(
        self,
        images_dir: str,
        case_list: Sequence[str],
        crop_size: Sequence[int],
        spacing: Sequence[float],
        num_samples: int,
        csv_file: str,
        blank_side: int = 0,
        lesion_label: Sequence[str] | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.case_list = list(case_list)
        self.csv_file = csv_file
        self.labels_df = _load_labels_csv(csv_file)
        self.csv_list = [
            self.labels_df[self.labels_df["seriesuid"] == case].reset_index(drop=True)
            for case in self.case_list
        ]
        self.lesion_label = list(lesion_label) if lesion_label is not None else DEFAULT_LESION_LABELS

        self.transform_post = torchvision.transforms.Compose(
            [
                transform.RandomFlip(flip_depth=True, flip_height=True, flip_width=True, p=0.5),
                transform.RandomTranspose(p=0.5, trans_xy=True, trans_zx=False, trans_zy=False),
                transform.Pad(output_size=list(crop_size)),
                transform.RandomCrop(output_size=list(crop_size), pos_ratio=0.9),
                transform.CoordToAnnot(blank_side=blank_side),
            ]
        )
        self.crop_fn = InstanceCrop(
            crop_size=list(crop_size),
            tp_ratio=0.75,
            spacing=list(spacing),
            rand_trans=[10, 20, 20],
            rand_rot=[20, 0, 0],
            rand_space=[0.9, 1.2],
            sample_num=num_samples,
            blank_side=blank_side,
            instance_crop=True,
        )

    def __len__(self):
        return len(self.case_list)

    def _volume_path(self, case: str):
        direct = self.images_dir / case / f"{case}_volume.nii.gz"
        if direct.exists():
            return str(direct)
        for subset in sorted(self.images_dir.glob("subset*")):
            candidate = subset / case / f"{case}_volume.nii.gz"
            if candidate.exists():
                return str(candidate)
        return str(direct)

    def __getitem__(self, idx):
        case = self.case_list[idx]
        image, image_spacing = _load_luna_volume(self._volume_path(case))
        csv_label = self.csv_list[idx]
        if csv_label.empty:
            all_loc = np.empty((0, 3), dtype="float32")
            all_rad = np.empty((0, 3), dtype="float32")
            labels = np.empty((0,), dtype=str)
        else:
            all_loc, all_rad = _labels_to_arrays(csv_label)
            labels = csv_label["label"].astype(str).str.lower().to_numpy()

        lesion_index = np.isin(labels, self.lesion_label)
        all_cls = np.ones(shape=(all_loc.shape[0]), dtype="int8") * (-1)
        all_cls[lesion_index] = 0

        samples = self.crop_fn(
            {
                "image": image,
                "all_loc": all_loc,
                "all_rad": all_rad,
                "all_cls": all_cls,
                "file_name": case,
            },
            image_spacing,
        )
        random_samples = []
        for sample in samples:
            sample = self.transform_post(sample)
            sample["image"] = sample["image"] * 2.0 - 1.0
            random_samples.append(sample)
        return random_samples


class Luna16CPMNetEvalDataset(Dataset):
    """LUNA16 NIfTI adapter for CPMNetv2 split-combine validation/test."""

    def __init__(
        self,
        images_dir: str,
        case_list: Sequence[str],
        splitcomb: SplitComb,
        csv_file: str,
    ):
        self.images_dir = Path(images_dir)
        self.case_list = list(case_list)
        self.splitcomb = splitcomb
        self.csv_file = csv_file
        self.labels_df = _load_labels_csv(csv_file)
        self.csv_list = [
            self.labels_df[self.labels_df["seriesuid"] == case].reset_index(drop=True)
            for case in self.case_list
        ]

    def __len__(self):
        return len(self.case_list)

    def _volume_path(self, case: str):
        direct = self.images_dir / case / f"{case}_volume.nii.gz"
        if direct.exists():
            return str(direct)
        for subset in sorted(self.images_dir.glob("subset*")):
            candidate = subset / case / f"{case}_volume.nii.gz"
            if candidate.exists():
                return str(candidate)
        return str(direct)

    def __getitem__(self, idx):
        case = self.case_list[idx]
        image, image_spacing = _load_luna_volume(self._volume_path(case))
        image = image * 2.0 - 1.0
        split_images, nzhw = self.splitcomb.split(image)
        return {
            "split_images": np.ascontiguousarray(split_images),
            "file_name": case,
            "nzhw": nzhw,
            "spacing": image_spacing,
        }

    def annotations_dataframe(self):
        rows = []
        seriesuids = []
        columns = ["seriesuid", "coordX", "coordY", "coordZ", "w", "h", "d"]
        for case, csv_label in zip(self.case_list, self.csv_list):
            seriesuids.append(case)
            for _, label in csv_label.iterrows():
                rows.append(
                    {
                        "seriesuid": case,
                        "coordX": label["x"],
                        "coordY": label["y"],
                        "coordZ": label["z"],
                        "w": label["w"],
                        "h": label["h"],
                        "d": label["d"],
                    }
                )
        return pd.DataFrame(rows, columns=columns), pd.DataFrame(seriesuids)
