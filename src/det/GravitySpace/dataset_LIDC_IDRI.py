import os
import sys
import random
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import binary_fill_holes

from src.det.GravitySpace.utility.msg.msg_error import msg_error


class LIDC_IDRI_volume(Dataset):
    """
    Dataset class that handles 3D LIDC-IDRI volumes and corresponding 3D annotation masks.
    Supports loading precomputed annotation centers and optionally precomputed slices for faster data loading.
    """

    def __init__(self,
                 images_dir: str,
                 annotations_dir: str,
                 case_list: list,
                 view: str,
                 transforms=None,
                 precomputed_centers_dir: str = None,
                 precomputed_slices_dir: str = None,
                 image_size: tuple = None,
                 use_subvolumes: bool = False,
                 subvolume_depth: int = 32,
                 subvolume_stride: int = 16,
                 val_subvolume_stride: int = None,
                 test_subvolume_stride: int = None,
                 positive_fraction: float = 0.7,
                 samples_per_epoch: int = None,
                 split: str = "train"):
        """
        :param images_dir: Directory containing the 3D LIDC-IDRI volumes (.nii.gz files).
        :param annotations_dir: Directory containing 3D annotation volumes (.nii.gz files).
        :param case_list: List of 3D LIDC-IDRI cases.
        :param view: View of the slices (axial, coronal, sagittal).
        :param transforms: Optional transforms to apply on the slices.
        :param precomputed_centers_dir: Optional directory with precomputed annotation centers (.npy files).
        :param precomputed_slices_dir: Optional directory with precomputed resized slices (.npz files).
        :param image_size: Optional tuple (height, width) for image size. Used to find correctly named .npy/.npz files.
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.case_list = case_list
        self.view = view
        self.transforms = transforms
        self.precomputed_centers_dir = precomputed_centers_dir
        self.precomputed_slices_dir = precomputed_slices_dir
        self.image_size = image_size
        self.use_subvolumes = use_subvolumes
        self.subvolume_depth = subvolume_depth
        self.subvolume_stride = subvolume_stride
        self.val_subvolume_stride = val_subvolume_stride
        self.test_subvolume_stride = test_subvolume_stride
        self.positive_fraction = positive_fraction
        self.samples_per_epoch = samples_per_epoch
        self.split = split
        
        # Check if precomputed centers are available
        if precomputed_centers_dir:
            self.use_precomputed_centers = os.path.isdir(precomputed_centers_dir)
            if self.use_precomputed_centers:
                size_suffix = f"_{image_size[0]}x{image_size[1]}" if image_size else ""
                #print(f"✅ Using precomputed annotation centers from {precomputed_centers_dir}{size_suffix}")
            else:
                print(f"⚠️  Precomputed centers dir not found: {precomputed_centers_dir}")
                print(f"   Will extract annotations at runtime (slower)")
                self.use_precomputed_centers = False
        else:
            self.use_precomputed_centers = False
        
        # Check if precomputed slices are available
        if precomputed_slices_dir:
            self.use_precomputed_slices = os.path.isdir(precomputed_slices_dir)
            if self.use_precomputed_slices:
                size_suffix = f"_{image_size[0]}x{image_size[1]}" if image_size else ""
                # print(f"✅ Using precomputed resized slices from {precomputed_slices_dir}{size_suffix}")
                # print(f"   💾 This skips Custom_Resize during training!")
            else:
                print(f"⚠️  Precomputed slices dir not found: {precomputed_slices_dir}")
                print(f"   Will resize slices at runtime (slower)")
                self.use_precomputed_slices = False
        else:
            self.use_precomputed_slices = False

        self.cache_case_data = self.use_subvolumes and self.split in ("val", "test")
        self.max_cached_cases = 2
        self._case_cache = OrderedDict()

        self.windows = []
        self.positive_windows = []
        self.negative_windows = []
        if self.use_subvolumes:
            self._build_subvolume_index()

    def __len__(self):
        if self.use_subvolumes:
            if self.split == "train":
                return self.samples_per_epoch or max(len(self.windows), len(self.case_list))
            return len(self.windows)
        return len(self.case_list)

    def _size_suffix(self):
        return f"_{self.image_size[0]}x{self.image_size[1]}" if self.image_size else ""

    def _load_slices(self, case):
        slices = None
        if self.use_precomputed_slices:
            precomputed_slices_path = os.path.join(
                self.precomputed_slices_dir, case,
                f"{case}_slices_{self.view}{self._size_suffix()}.npy"
            )
            if os.path.exists(precomputed_slices_path):
                slices = np.load(precomputed_slices_path)
            else:
                print(f"⚠️  Precomputed slices not found for {precomputed_slices_path}, will load raw slices")

        if slices is None:
            image_path = os.path.join(self.images_dir, case, case + '_volume.nii.gz')
            image = nib.load(image_path).get_fdata()  # Shape: (X, Y, Z)

            if self.view == "axial":
                slices = image.transpose(2, 0, 1)  # (Z, X, Y) for Z slices
            elif self.view == "coronal":
                slices = image.transpose(1, 0, 2)  # (Y, X, Z) for Y slices
            elif self.view == "sagittal":
                slices = image.transpose(0, 1, 2)  # (X, Y, Z) for X slices
            else:
                str_err = msg_error(file=__file__,
                                    variable=self.view,
                                    type_variable="view",
                                    choices="[axial, coronal, sagittal]")
                sys.exit(str_err)

        return slices

    def _load_annotations(self, case):
        annotation = None
        if self.use_precomputed_centers:
            precomputed_path = os.path.join(
                self.precomputed_centers_dir,
                case,
                f"{case}_annotation_centers_{self.view}{self._size_suffix()}.npy"
            )
            if os.path.exists(precomputed_path):
                annotation = np.load(precomputed_path)
            else:
                print(f"⚠️  Precomputed centers not found for {case}, will extract at runtime")

        if annotation is None:
            annotation_path = os.path.join(self.annotations_dir, case, case + '_nodule_mask.nii.gz')
            annotation = nib.load(annotation_path).get_fdata()

            if self.view == "axial":
                annotation = annotation.transpose(2, 0, 1)  # (Z, X, Y) for Z slices
            elif self.view == "coronal":
                annotation = annotation.transpose(1, 0, 2)  # (Y, X, Z) for Y slices
            elif self.view == "sagittal":
                annotation = annotation.transpose(0, 1, 2)  # (X, Y, Z) for X slices
            else:
                str_err = msg_error(file=__file__,
                                    variable=self.view,
                                    type_variable="view",
                                    choices="[axial, coronal, sagittal]")
                sys.exit(str_err)
            
            # Normalize annotation to 1 channel [S, H, W]
            if annotation.ndim == 4:
                # [S, C, H, W] -> take first channel or merge
                annotation = annotation[:, 0, :, :]  # [S, H, W]
            # If already 3D [S, H, W], keep as is

        return annotation

    def _load_case_data(self, case):
        if not self.cache_case_data:
            return self._load_slices(case), self._load_annotations(case)

        if case in self._case_cache:
            self._case_cache.move_to_end(case)
            return self._case_cache[case]

        case_data = (self._load_slices(case), self._load_annotations(case))
        self._case_cache[case] = case_data
        self._case_cache.move_to_end(case)

        while len(self._case_cache) > self.max_cached_cases:
            self._case_cache.popitem(last=False)

        return case_data

    def _annotation_has_nodule(self, annotation):
        if annotation.ndim == 3 and annotation.shape[-1] == 4:
            return np.any(annotation[:, :, 0] != -1)
        return np.any(annotation > 0)

    def _make_windows_for_case(self, case, num_slices, annotation):
        depth = min(self.subvolume_depth, num_slices)
        stride = max(1, self.subvolume_stride)

        if self.split == "train":
            starts = list(range(0, max(num_slices - depth + 1, 1), stride))
            last_start = max(num_slices - depth, 0)
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            eval_ranges = [(start, min(start + depth, num_slices)) for start in starts]
        elif self.split in ("val", "test") and (
            (self.split == "val" and self.val_subvolume_stride is not None) or
            (self.split == "test" and self.test_subvolume_stride is not None)
        ):
            eval_stride_cfg = self.val_subvolume_stride if self.split == "val" else self.test_subvolume_stride
            eval_stride = min(max(1, eval_stride_cfg), depth)
            starts = list(range(0, max(num_slices - depth + 1, 1), eval_stride))
            last_start = max(num_slices - depth, 0)
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            starts = sorted(set(starts))
            eval_ranges = [
                (start, starts[idx + 1] if idx + 1 < len(starts) else num_slices)
                for idx, start in enumerate(starts)
            ]
        else:
            starts = list(range(0, num_slices, depth))
            eval_ranges = [(start, min(start + depth, num_slices)) for start in starts]

        windows = []
        for start, (eval_start, eval_end) in zip(starts, eval_ranges):
            end = min(start + depth, num_slices)
            ann_window = annotation[start:end]
            windows.append({
                "case": case,
                "start": start,
                "end": end,
                "eval_start": eval_start,
                "eval_end": eval_end,
                "has_nodule": self._annotation_has_nodule(ann_window),
            })

        return windows

    def _pad_subvolume(self, slices, annotation):
        current_depth = slices.shape[0]
        pad_amount = self.subvolume_depth - current_depth
        if pad_amount <= 0:
            return slices, annotation

        slices_pad = np.zeros((pad_amount, *slices.shape[1:]), dtype=slices.dtype)
        slices = np.concatenate([slices, slices_pad], axis=0)

        if annotation.ndim == 3 and annotation.shape[-1] == 4:
            annotation_pad = np.full((pad_amount, *annotation.shape[1:]), -1, dtype=annotation.dtype)
        else:
            annotation_pad = np.zeros((pad_amount, *annotation.shape[1:]), dtype=annotation.dtype)
        annotation = np.concatenate([annotation, annotation_pad], axis=0)

        return slices, annotation

    def _build_subvolume_index(self):
        for case in self.case_list:
            annotation = self._load_annotations(case)
            num_slices = annotation.shape[0]
            case_windows = self._make_windows_for_case(case, num_slices, annotation)
            self.windows.extend(case_windows)

        self.positive_windows = [w for w in self.windows if w["has_nodule"]]
        self.negative_windows = [w for w in self.windows if not w["has_nodule"]]

        print(
            f"Subvolume index ({self.split}): {len(self.windows)} windows "
            f"({len(self.positive_windows)} positive, {len(self.negative_windows)} negative)"
        )

    def _select_train_window(self):
        use_positive = random.random() < self.positive_fraction
        if use_positive and self.positive_windows:
            return random.choice(self.positive_windows)
        if self.negative_windows:
            return random.choice(self.negative_windows)
        if self.positive_windows:
            return random.choice(self.positive_windows)
        return self.windows[0]

    def __getitem__(self, idx):

        if self.use_subvolumes:
            window = self._select_train_window() if self.split == "train" else self.windows[idx]
            case = window["case"]
            start = window["start"]
            end = window["end"]
            eval_start = window["eval_start"]
            eval_end = window["eval_end"]
        else:
            case = self.case_list[idx]
            start = None
            end = None
            eval_start = None
            eval_end = None

        slices, annotation = self._load_case_data(case)

        if self.use_subvolumes:
            slices = slices[start:end]
            annotation = annotation[start:end]
            slice_indices = range(start, end)
            original_length = end - start
            eval_slice_start = eval_start - start
            eval_slice_end = eval_end - start
            slices, annotation = self._pad_subvolume(slices, annotation)
        else:
            slice_indices = range(slices.shape[0])
            original_length = slices.shape[0]
            eval_slice_start = 0
            eval_slice_end = original_length

        slicenames = ["{}_{}_slice_{}".format(case, self.view, str(i + 1).zfill(3)) for i in slice_indices]
        if self.use_subvolumes and len(slicenames) < slices.shape[0]:
            slicenames.extend([f"{case}_{self.view}_pad_{i + 1:03d}" for i in range(slices.shape[0] - len(slicenames))])

        sample = {
            'case': case,
            'slices': slices,
            # 'masks': masks,
            'annotations': annotation,
            'slicenames': slicenames,
            'original_length': original_length
        }

        if self.use_subvolumes:
            sample['subvolume_start'] = start
            sample['subvolume_end'] = end
            sample['eval_slice_start'] = eval_slice_start
            sample['eval_slice_end'] = eval_slice_end
            sample['has_nodule'] = window["has_nodule"]

        # Tranforms
        if self.transforms:
            sample = self.transforms(sample)

        return sample
