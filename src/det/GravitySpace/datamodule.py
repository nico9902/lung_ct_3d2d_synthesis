import os
import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

# Add project root to sys.path for local imports
import sys
sys.path.append(os.getcwd())

from src.det.GravitySpace.dataset_LIDC_IDRI import LIDC_IDRI_volume
from src.det.GravitySpace.transforms import Custom_Resize, CustomToTensor, RepeatChannels, AdjacentSliceChannels, ExtractAnnotationCenter

def collate_fn(batch):
    """
    Optimized collate function for handling variable-length volumes.
    
    ⚡ CRITICAL OPTIMIZATION:
    - Conversion to tensor happens in transforms.CustomToTensor 
    - Padding uses direct torch.pad (no permute→pad→permute)
    - Only one memory allocation per operation
    
    Batch structure:
      slices:      [S, C, H, W] (tensor or numpy, floating-point)
      annotations: [S, H, W]    (tensor or numpy, floating-point)
    """
    # Extract from batch
    slices_list = [b["slices"] for b in batch]
    annotations_list = [b["annotations"] for b in batch]
    
    # Ensure slices are tensors (should be from CustomToTensor)
    slices_list = [torch.from_numpy(s).float() if isinstance(s, np.ndarray) else s 
                   for s in slices_list]
    
    # Ensure annotations are tensors
    annotations_list = [torch.from_numpy(a).float() if isinstance(a, np.ndarray) else a 
                        for a in annotations_list]
    
    # ⚡ FAST: Find max_slices
    max_slices = max([s.shape[0] for s in slices_list])
    
    # ⚡ FAST: Direct padding without permute (contiguous memory layout)
    padded_slices = []
    padded_annotations = []
    
    for slices, annotations in zip(slices_list, annotations_list):
        num_slices = slices.shape[0]
        pad_amount = max_slices - num_slices
        
        if pad_amount > 0:
            # Pad on first dimension (slices) with zeros
            # torch.pad([S, C, H, W], (0, 0, 0, 0, 0, 0, 0, pad_amount)) → [S+pad, C, H, W]
            slices = torch.nn.functional.pad(slices.unsqueeze(0), 
                                           (0, 0, 0, 0, 0, 0, 0, pad_amount)).squeeze(0)
            annotation_pad_value = -1 if annotations.dim() == 3 and annotations.shape[-1] == 4 else 0
            annotations = torch.nn.functional.pad(annotations.unsqueeze(0),
                                                 (0, 0, 0, 0, 0, pad_amount),
                                                 value=annotation_pad_value).squeeze(0)
        
        padded_slices.append(slices)
        padded_annotations.append(annotations)
    
    # ⚡ FAST: Stack all at once
    slices_out = torch.stack(padded_slices, dim=0)           # [B, S, C, H, W]
    annotations_out = torch.stack(padded_annotations, dim=0)  # [B, S, H, W]
    
    slicenames = [b["slicenames"] for b in batch]
    case = [b["case"] for b in batch]
    original_lengths = torch.tensor([
        b.get("original_length", s.shape[0])
        for b, s in zip(batch, slices_list)
    ])

    output = {
        "slices": slices_out,
        "annotations": annotations_out,
        "slicenames": slicenames,
        "case": case,
        "original_lengths": original_lengths
    }

    if "subvolume_start" in batch[0]:
        output["subvolume_start"] = torch.tensor([b["subvolume_start"] for b in batch])
        output["subvolume_end"] = torch.tensor([b["subvolume_end"] for b in batch])
        output["eval_slice_start"] = torch.tensor([b["eval_slice_start"] for b in batch])
        output["eval_slice_end"] = torch.tensor([b["eval_slice_end"] for b in batch])
        output["has_nodule"] = torch.tensor([bool(b["has_nodule"]) for b in batch])

    return output

class LIDC_DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for LIDC-IDRI 3D detection volumes.
    Supports loading precomputed annotation centers for faster data loading.
    """
    def __init__(self, 
                 images_dir: str, 
                 annotations_dir: str, 
                 train_cases: list, 
                 val_cases: list,
                 test_cases: list, 
                 batch_size: int = 1, 
                 image_size: tuple = (352, 480),
                 view: str = "axial",
                 num_workers: int = 4,
                 precomputed_centers_dir: str = None,
                 precomputed_slices_dir: str = None,
                 use_subvolumes: bool = False,
                 subvolume_depth: int = 32,
                 subvolume_stride: int = 16,
                 val_subvolume_stride: int = None,
                 test_subvolume_stride: int = None,
                 positive_fraction: float = 0.7,
                 samples_per_epoch: int = None,
                 input_mode: str = "2d",
                 context_slices: int = 3):
        super().__init__()
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.train_cases = train_cases
        self.val_cases = val_cases
        self.test_cases = test_cases
        self.batch_size = batch_size
        self.image_size = image_size  # Store for passing to datasets
        self.view = view
        self.num_workers = num_workers
        self.precomputed_centers_dir = precomputed_centers_dir
        self.precomputed_slices_dir = precomputed_slices_dir
        self.use_subvolumes = use_subvolumes
        self.subvolume_depth = subvolume_depth
        self.subvolume_stride = subvolume_stride
        self.val_subvolume_stride = val_subvolume_stride
        self.test_subvolume_stride = test_subvolume_stride
        self.positive_fraction = positive_fraction
        self.samples_per_epoch = samples_per_epoch
        self.input_mode = input_mode
        self.context_slices = context_slices

        if self.input_mode not in ("2d", "2.5d"):
            raise ValueError("input_mode must be either '2d' or '2.5d'")
        if self.input_mode == "2.5d" and self.context_slices != 3:
            raise ValueError("2.5D input supports exactly 3 channels: slice i-1, slice i, slice i+1")
        channel_transform = (
            AdjacentSliceChannels()
            if self.input_mode == "2.5d"
            else RepeatChannels(repeats=3)
        )
        
        # Setup Transforms
        if precomputed_slices_dir and os.path.isdir(precomputed_slices_dir) and precomputed_centers_dir and os.path.isdir(precomputed_centers_dir):
            print(f"⚠️  Precomputed slices provided, skipping Custom_Resize and ExtractAnnotationCenter in transforms!")
            transform_list = [
                CustomToTensor(),
                channel_transform
            ]
        elif precomputed_slices_dir and os.path.isdir(precomputed_slices_dir):
            print(f"⚠️  Precomputed slices provided, skipping Custom_Resize in transforms!")
            transform_list = [
                CustomToTensor(),
                ExtractAnnotationCenter(),
                channel_transform
            ]
        else:
            print(f"⚠️  No precomputed slices provided, using full transforms (Custom_Resize and ExtractAnnotationCenter)!")
            transform_list = [
                Custom_Resize(size=image_size),
                CustomToTensor(),
                ExtractAnnotationCenter(),
                channel_transform
            ]
        self.transforms = Compose(transform_list)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = LIDC_IDRI_volume(
                images_dir=self.images_dir,
                annotations_dir=self.annotations_dir,
                case_list=self.train_cases,
                view=self.view,
                transforms=self.transforms,
                precomputed_centers_dir=self.precomputed_centers_dir,
                precomputed_slices_dir=self.precomputed_slices_dir,
                image_size=self.image_size,
                use_subvolumes=self.use_subvolumes,
                subvolume_depth=self.subvolume_depth,
                subvolume_stride=self.subvolume_stride,
                val_subvolume_stride=self.val_subvolume_stride,
                test_subvolume_stride=self.test_subvolume_stride,
                positive_fraction=self.positive_fraction,
                samples_per_epoch=self.samples_per_epoch,
                split="train"
            )
            self.val_ds = LIDC_IDRI_volume(
                images_dir=self.images_dir,
                annotations_dir=self.annotations_dir,
                case_list=self.val_cases,
                view=self.view,
                transforms=self.transforms,
                precomputed_centers_dir=self.precomputed_centers_dir,
                precomputed_slices_dir=self.precomputed_slices_dir,
                image_size=self.image_size,
                use_subvolumes=self.use_subvolumes,
                subvolume_depth=self.subvolume_depth,
                subvolume_stride=self.subvolume_stride,
                val_subvolume_stride=self.val_subvolume_stride,
                test_subvolume_stride=self.test_subvolume_stride,
                positive_fraction=self.positive_fraction,
                samples_per_epoch=None,
                split="val"
            )
        if stage == 'test' or stage is None:
            self.test_ds = LIDC_IDRI_volume(
                images_dir=self.images_dir,
                annotations_dir=self.annotations_dir,
                case_list=self.test_cases,
                view=self.view,
                transforms=self.transforms,
                precomputed_centers_dir=self.precomputed_centers_dir,
                precomputed_slices_dir=self.precomputed_slices_dir,
                image_size=self.image_size,
                use_subvolumes=self.use_subvolumes,
                subvolume_depth=self.subvolume_depth,
                subvolume_stride=self.subvolume_stride,
                val_subvolume_stride=self.val_subvolume_stride,
                test_subvolume_stride=self.test_subvolume_stride,
                positive_fraction=self.positive_fraction,
                samples_per_epoch=None,
                split="test"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=4,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=4,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=4,
            persistent_workers=True if self.num_workers > 0 else False
        )
