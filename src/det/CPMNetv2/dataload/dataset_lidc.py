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


VALID_VIEWS = ("axial", "coronal", "sagittal")
LABEL_COLUMNS = ["seriesuid", "x", "y", "z", "w", "h", "d", "label"]
DEFAULT_LESION_LABELS = ["nodule"]

# Each view maps normal CSV box columns into the tensor axis order returned by
# _load_lidc_volume. Axial is the CPMNet-native (z, y, x) convention.
VIEW_AXIS_COLUMNS = {
    "axial": (["z", "y", "x"], ["d", "h", "w"]),
    "coronal": (["y", "x", "z"], ["h", "w", "d"]),
    "sagittal": (["x", "y", "z"], ["w", "h", "d"]),
}

def _validate_view(view: str) -> str:
    if view not in VALID_VIEWS:
        raise ValueError(f"view must be one of: {', '.join(VALID_VIEWS)}")
    return view


def _norm_hu(data: np.ndarray, min_value: float = -200.0, max_value: float = 800.0) -> np.ndarray:
    data = data.astype("float32", copy=False)
    data = np.clip(data, min_value, max_value)
    return (data - min_value) / (max_value - min_value)


def _load_lidc_volume(path: str, view: str):
    view = _validate_view(view)
    nii = nib.load(path)
    volume = nii.get_fdata().astype("float32")
    zoom_x, zoom_y, zoom_z = nii.header.get_zooms()[:3]

    if view == "axial":
        # NIfTI is read as (x, y, z). CPMNet expects axial tensors as (z, y, x).
        volume = volume.transpose(2, 1, 0)
        spacing = np.array([zoom_z, zoom_y, zoom_x], dtype="float32")
    elif view == "coronal":
        volume = volume.transpose(1, 0, 2)
        spacing = np.array([zoom_y, zoom_x, zoom_z], dtype="float32")
    elif view == "sagittal":
        volume = volume.transpose(0, 1, 2)
        spacing = np.array([zoom_x, zoom_y, zoom_z], dtype="float32")

    return volume, spacing


def _empty_label_frame():
    return pd.DataFrame(columns=LABEL_COLUMNS)


def _load_labels_csv(csv_file: str):
    if csv_file is None:
        return _empty_label_frame()
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"LIDC label CSV not found: {csv_file}")

    labels = pd.read_csv(csv_file)
    missing = [column for column in LABEL_COLUMNS if column not in labels.columns]
    if missing:
        raise ValueError(f"LIDC label CSV is missing columns: {missing}")
    labels = labels[LABEL_COLUMNS]
    labels = labels[labels["label"].astype(str).str.lower().isin(DEFAULT_LESION_LABELS)]
    labels = labels.dropna(subset=["x", "y", "z", "w", "h", "d"])
    return labels


def _labels_to_view_arrays(csv_label: pd.DataFrame, view: str):
    view = _validate_view(view)
    loc_columns, size_columns = VIEW_AXIS_COLUMNS[view]
    all_loc = csv_label[loc_columns].to_numpy(dtype="float32")
    all_rad = csv_label[size_columns].to_numpy(dtype="float32")
    return all_loc, all_rad

class LIDCCPMNetTrainDataset(Dataset):
    """LIDC-IDRI adapter that feeds CPMNetv2's original training crop pipeline."""

    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        case_list: Sequence[str],
        view: str,
        crop_size: Sequence[int],
        spacing: Sequence[float],
        num_samples: int,
        csv_file: str = None,
        blank_side: int = 0,
        lesion_label: Sequence[str] = None,
    ):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.case_list = list(case_list)
        self.csv_file = csv_file or os.path.join(annotations_dir, "lidc_labels.csv")
        self.labels_df = _load_labels_csv(self.csv_file)
        self.csv_list = [
            self.labels_df[self.labels_df["seriesuid"] == case].reset_index(drop=True)
            for case in self.case_list
        ]
        self.view = _validate_view(view)

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

        if lesion_label is None:
            self.lesion_label = DEFAULT_LESION_LABELS
        else:
            self.lesion_label = lesion_label

    def __len__(self):
        return len(self.case_list)

    def _paths(self, case: str):
        image_path = os.path.join(self.images_dir, case, f"{case}_volume.nii.gz")
        if os.path.exists(image_path):
            return image_path
        for subset in sorted(Path(self.images_dir).glob("subset*")):
            candidate = subset / case / f"{case}_volume.nii.gz"
            if candidate.exists():
                return str(candidate)
        return image_path

    def __getitem__(self, idx):
        case = self.case_list[idx]
        image_path = self._paths(case)
        image, image_spacing = _load_lidc_volume(image_path, self.view)
        image = _norm_hu(image)
        csv_label = self.csv_list[idx]
        if csv_label.empty:
            all_loc = np.empty((0, 3), dtype="float32")
            all_rad = np.empty((0, 3), dtype="float32")
            labels = np.empty((0,), dtype=str)
        else:
            all_loc, all_rad = _labels_to_view_arrays(csv_label, self.view)
            labels = csv_label["label"].astype(str).to_numpy()
        lesion_index = np.isin(labels, self.lesion_label)
        all_cls = np.ones(shape=(all_loc.shape[0]), dtype="int8") * (-1)
        all_cls[lesion_index] = 0

        data = {
            "image": image,
            "all_loc": all_loc,
            "all_rad": all_rad,
            "all_cls": all_cls,
            "file_name": case,
        }
        samples = self.crop_fn(data, image_spacing)
        random_samples = []
        for sample in samples:
            sample = self.transform_post(sample)
            sample["image"] = sample["image"] * 2.0 - 1.0   # Normalize to [-1, 1]
            random_samples.append(sample)
        return random_samples


class LIDCCPMNetEvalDataset(Dataset):
    """LIDC-IDRI adapter for CPMNetv2's original split-combine validation/test path."""

    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        case_list: Sequence[str],
        view: str,
        splitcomb: SplitComb,
        csv_file: str = None,
    ):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.case_list = list(case_list)
        self.view = _validate_view(view)
        self.splitcomb = splitcomb
        self.csv_file = csv_file or os.path.join(annotations_dir, "lidc_labels.csv")
        self.labels_df = _load_labels_csv(self.csv_file)
        self.csv_list = [
            self.labels_df[self.labels_df["seriesuid"] == case].reset_index(drop=True)
            for case in self.case_list
        ]

    def __len__(self):
        return len(self.case_list)

    def _paths(self, case: str):
        image_path = os.path.join(self.images_dir, case, f"{case}_volume.nii.gz")
        if os.path.exists(image_path):
            return image_path
        for subset in sorted(Path(self.images_dir).glob("subset*")):
            candidate = subset / case / f"{case}_volume.nii.gz"
            if candidate.exists():
                return str(candidate)
        return image_path

    def __getitem__(self, idx):
        case = self.case_list[idx]
        image_path = self._paths(case)
        image, image_spacing = _load_lidc_volume(image_path, self.view)
        image = _norm_hu(image) * 2.0 - 1.0
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
