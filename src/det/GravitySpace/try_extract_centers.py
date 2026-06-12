import os
import sys
import numpy as np
import torch
import nibabel as nib

# Add project root to path
sys.path.append(os.getcwd())

from src.det.GravitySpace.dataset_LIDC_IDRI import LIDC_IDRI_volume
from src.det.GravitySpace.transforms import ExtractAnnotationCenter

def try_extract_centers():
    images_dir = "data/LIDC-IDRI_nifti_z_only"
    annotations_dir = "data/LIDC-IDRI_nifti_z_only"
    case = "LIDC-IDRI-0003"
    view = "axial"
    
    # Initialize dataset with the specific transform
    transform = ExtractAnnotationCenter(max_nodules=10)
    dataset = LIDC_IDRI_volume(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        case_list=[case],
        view=view,
        transforms=transform
    )
    
    print(f"Loading case: {case}")
    sample = dataset[0]
    
    print(f"Sample keys: {sample.keys()}")
    print(f"Slices shape: {sample['slices'].shape}")
    print(f"Annotations shape: {sample['annotations'].shape}")
    
    # Check if we have any valid annotations
    annotations = sample['annotations']
    slicenames = sample['slicenames']
    valid_mask = annotations[:, :, 0] != -1
    num_nodules_per_slice = valid_mask.sum(axis=1)
    
    total_nodules_found = valid_mask.sum()
    print(f"Total nodule occurrences across slices: {total_nodules_found}")
    
    # Show some examples
    slices_with_nodules = np.where(num_nodules_per_slice > 0)[0]
    if len(slices_with_nodules) > 0:
        print(f"Slices with nodules: {slices_with_nodules}")
        for s_idx in slices_with_nodules: # Show first 5 slices with nodules
            print(f"Slice {s_idx}:")
            print(f"  Slicename: {slicenames[s_idx]}")
            for n_idx in range(10):
                center = annotations[s_idx, n_idx]
                if center[0] != -1:
                    print(f"  Nodule {n_idx}: center=({center[0]:.2f}, {center[1]:.2f}), radius=({center[2]:.2f}, {center[3]:.2f})")
    else:
        print("No nodules found in this case.")

if __name__ == "__main__":
    try_extract_centers()
