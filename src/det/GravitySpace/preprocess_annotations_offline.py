"""
Precompute annotation centers offline instead of extracting them during training.
This is a ONE-TIME operation that should be run before training.

Saves extracted centers as .npy files to avoid repeated computation.
Expected speedup: 50-80% reduction in dataloader time.
"""

import os
import numpy as np
import nibabel as nib
import cv2
from scipy.ndimage import label
from scipy.spatial.distance import cdist
from pathlib import Path
import argparse
from tqdm import tqdm
import time

def min_dist(a, b):
    """Compute minimum distance between two point clouds."""
    return np.min(cdist(a, b))

def merge_components(components, thresh=2.0):
    """Merge connected components that are closer than threshold."""
    components = components.copy()
    
    while True:
        n = len(components)
        if n <= 1:
            break
        
        best_i, best_j = -1, -1
        best_d = np.inf
        
        for i in range(n):
            for j in range(i + 1, n):
                d = min_dist(components[i], components[j])
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j
        
        if best_d > thresh:
            break
        
        components[best_i] = np.vstack([
            components[best_i],
            components[best_j]
        ])
        components.pop(best_j)
    
    return components

def resize_volume_slices(image_volume, image_size):
    """
    Resize a 3D image volume using OpenCV (INTER_AREA for best quality).
    
    Args:
        image_volume: [Z, H, W] image volume
        image_size: tuple (height, width) or list [height, width]
    
    Returns:
        resized_volume: [Z, new_H, new_W] resized image volume
    """
    size = tuple(image_size) if isinstance(image_size, list) else image_size
    Z = image_volume.shape[0]
    resized_slices = []
    
    for z in range(Z):
        slice_z = image_volume[z]  # [H, W]
        # Use INTER_AREA for better quality downsampling
        resized_slice = cv2.resize(slice_z, size, interpolation=cv2.INTER_AREA)
        resized_slices.append(resized_slice)
    
    return np.stack(resized_slices, axis=0)

def resize_annotation_volume(annotation_volume, image_size):
    """
    Resize a 3D annotation volume using OpenCV.
    
    Args:
        annotation_volume: [Z, H, W] mask volume
        image_size: tuple (height, width) or list [height, width]
    
    Returns:
        resized_volume: [Z, new_H, new_W] resized mask volume
    """
    size = tuple(image_size) if isinstance(image_size, list) else image_size
    Z = annotation_volume.shape[0]
    resized_slices = []
    
    for z in range(Z):
        slice_z = annotation_volume[z]  # [H, W]
        # Use INTER_NEAREST to preserve exact label values
        resized_slice = cv2.resize(slice_z, size, interpolation=cv2.INTER_NEAREST)
        resized_slices.append(resized_slice)
    
    return np.stack(resized_slices, axis=0)

def extract_annotation_centers(annotation_volume, max_nodules=10, image_size=None):
    """
    Extract annotation centers from a 3D volume.
    
    Args:
        annotation_volume: [Z, H, W] mask volume
        max_nodules: Max nodules per slice
        image_size: Optional tuple (height, width) to resize before extraction
    
    Returns:
        centers: [Z, max_nodules, 4] array where each row is [cx, cy, rx, ry]
    """
    # Optionally resize volume before extraction
    if image_size is not None:
        annotation_volume = resize_annotation_volume(annotation_volume, image_size)
    
    # Connected Components 3D
    labeled, num = label(annotation_volume)
    
    components = [
        np.argwhere(labeled == i)
        for i in range(1, num + 1)
    ]
    
    # Merge nearby components
    components = merge_components(components, thresh=2.0)
    
    # Extract centers per slice
    def extract_from_slice(components, z):
        centers = np.full((max_nodules, 4), -1, dtype=np.float32)
        
        slice_components = []
        for comp in components:
            if np.any(comp[:, 0] == z):
                slice_components.append(comp)
        
        for i, comp in enumerate(slice_components[:max_nodules]):
            coords = comp[comp[:, 0] == z]
            
            if len(coords) == 0:
                continue
            
            y = coords[:, 1]
            x = coords[:, 2]
            
            center_x = np.mean(x)
            center_y = np.mean(y)
            radius_x = (np.max(x) - np.min(x)) / 2.0
            radius_y = (np.max(y) - np.min(y)) / 2.0
            
            centers[i] = [center_x, center_y, radius_x, radius_y]
        
        return centers
    
    # Apply per slice
    centers = np.array([
        extract_from_slice(components, z)
        for z in range(annotation_volume.shape[0])
    ], dtype=np.float32)
    
    return centers

def preprocess_annotations(annotations_dir, output_dir, images_dir=None, view="axial", image_size=None, save_slices=False):
    """
    Preprocess all annotation volumes in a directory.
    
    Args:
        annotations_dir: Directory containing *_nodule_mask.nii.gz files
        output_dir: Directory to save .npy files
        images_dir: Optional directory with image volumes (.nii.gz). Required if save_slices=True
        view: Which view to process (axial, coronal, sagittal)
        image_size: Optional tuple/list (height, width) to resize annotations before extraction
        save_slices: If True, also saves resized image slices as .npz files (requires images_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if save_slices and not images_dir:
        print("❌ Error: images_dir is required when save_slices=True")
        return
    
    # Find all annotation files
    annotation_files = sorted(Path(annotations_dir).glob("*/LIDC-IDRI-*_nodule_mask.nii.gz"))
    
    if not annotation_files:
        annotation_files = sorted(Path(annotations_dir).glob("*/*_nodule_mask.nii.gz"))
    
    print(f"Found {len(annotation_files)} annotation files")
    if save_slices:
        print(f"Will also save resized image slices from {images_dir}")
    
    times_centers = []
    times_slices = []
    
    for annotation_path in tqdm(annotation_files, desc="Processing annotations"):
        case_name = annotation_path.parent.name
        
        # Include image size in filename if resizing
        size_suffix = f"_{image_size[0]}x{image_size[1]}" if image_size else ""
        centers_output_path = os.path.join(output_dir, case_name, f"{case_name}_annotation_centers_{view}{size_suffix}.npy")
        slices_output_path = os.path.join(output_dir, case_name, f"{case_name}_slices_{view}{size_suffix}.npy") if save_slices else None
        
        # Skip if already processed
        centers_exists = os.path.exists(centers_output_path)
        slices_exists = os.path.exists(slices_output_path) if save_slices else True
        
        if centers_exists and slices_exists:
            print(f"⏭️  Skipping {case_name} (already exists)")
            continue
        
        # Load annotation
        annotation = nib.load(annotation_path).get_fdata()
        
        # Load image slices if needed
        image_slices = None
        if save_slices and not slices_exists:
            image_path = os.path.join(images_dir, case_name, case_name + '_volume.nii.gz')
            if os.path.exists(image_path):
                image_slices = nib.load(image_path).get_fdata()
            else:
                print(f"⚠️  Image file not found for {case_name}: {image_path}")
        
        # Apply view transpose
        if view == "axial":
            annotation = annotation.transpose(2, 0, 1)  # (Z, X, Y) for Z slices
            if image_slices is not None:
                image_slices = image_slices.transpose(2, 0, 1)
        elif view == "coronal":
            annotation = annotation.transpose(1, 0, 2)  # (Y, X, Z) for Y slices
            if image_slices is not None:
                image_slices = image_slices.transpose(1, 0, 2)
        elif view == "sagittal":
            annotation = annotation.transpose(0, 1, 2)  # (X, Y, Z) for X slices
            if image_slices is not None:
                image_slices = image_slices.transpose(0, 1, 2)
        
        # Normalize annotation to 1 channel
        if annotation.ndim == 4:
            annotation = annotation[:, 0, :, :]
        
        # Extract centers (with optional resizing)
        if not centers_exists:
            start = time.time()
            centers = extract_annotation_centers(annotation, max_nodules=10, image_size=image_size)
            print(f"✅ Extracted centers for {case_name}, shape: {centers.shape}, annotation shape: {annotation.shape}")
            elapsed = time.time() - start
            times_centers.append(elapsed)
            
            # Save centers
            os.makedirs(os.path.dirname(centers_output_path), exist_ok=True)
            np.save(centers_output_path, centers)
            print(f"✅ Saved centers {case_name}: {elapsed:.2f}s")
        
        # Save resized image slices if needed
        if save_slices and not slices_exists and image_slices is not None:
            start = time.time()
            if image_size:
                slices_resized = resize_volume_slices(image_slices, image_size)
            else:
                slices_resized = image_slices
            elapsed = time.time() - start
            times_slices.append(elapsed)
            
            # Save as compressed .npz (more efficient than .npy)
            os.makedirs(os.path.dirname(slices_output_path), exist_ok=True)
            np.save(slices_output_path, slices_resized.astype(np.float32))
            print(f"✅ Saved slices {case_name}: {elapsed:.2f}s")
    
    if times_centers:
        print(f"\n📊 Centers Statistics:")
        print(f"  Mean time per volume: {np.mean(times_centers):.2f}s")
        print(f"  Total time: {np.sum(times_centers):.1f}s")
        print(f"  Total speedup per epoch: {len(annotation_files) * np.mean(times_centers):.1f}s saved")
    
    if times_slices:
        print(f"\n📊 Slices Statistics:")
        print(f"  Mean time per volume: {np.mean(times_slices):.2f}s")
        print(f"  Total time: {np.sum(times_slices):.1f}s")
        print(f"  Total speedup per epoch: {len(annotation_files) * np.mean(times_slices):.1f}s saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute annotation centers and optionally resized slices offline")
    parser.add_argument("--annotations-dir", type=str, 
                        default="/ssd2/domenico/datasets/LIDC-IDRI_nifti",
                        help="Directory containing annotation volumes")
    parser.add_argument("--images-dir", type=str, 
                        default=None,
                        help="Directory containing image volumes (required if --save-slices is used)")
    parser.add_argument("--output-dir", type=str,
                        default="data/annotation_centers_cache",
                        help="Directory to save computed centers and slices")
    parser.add_argument("--view", type=str, default="axial",
                        choices=["axial", "coronal", "sagittal"],
                        help="View to process")
    parser.add_argument("--image-size", type=str, default=None,
                        help="Image size to resize annotations (format: 'height,width' or 'height x width'). "
                             "If not specified, uses original size. Example: '352,480' or '352x480'")
    parser.add_argument("--save-slices", action="store_true",
                        help="Also save resized image slices as .npz files (requires --images-dir)")
    
    args = parser.parse_args()
    
    # Parse image size if provided
    image_size = None
    if args.image_size:
        # Handle both comma and 'x' formats
        size_str = args.image_size.replace('x', ',').replace('X', ',')
        parts = size_str.split(',')
        if len(parts) == 2:
            try:
                image_size = (int(parts[0].strip()), int(parts[1].strip()))
            except ValueError:
                print(f"❌ Invalid image size format: {args.image_size}")
                print(f"   Use format: height,width (e.g., '352,480')")
                exit(1)
        else:
            print(f"❌ Invalid image size format: {args.image_size}")
            print(f"   Use format: height,width (e.g., '352,480')")
            exit(1)
    
    print(f"🔄 Preprocessing annotations...")
    print(f"  Input annotations:  {args.annotations_dir}")
    if args.save_slices:
        print(f"  Input images:       {args.images_dir}")
    print(f"  Output:             {args.output_dir}")
    print(f"  View:               {args.view}")
    print(f"  Image size:         {image_size if image_size else 'Original (no resize)'}")
    print(f"  Save slices:        {'Yes' if args.save_slices else 'No'}\n")
    
    preprocess_annotations(
        args.annotations_dir, 
        args.output_dir, 
        images_dir=args.images_dir,
        view=args.view, 
        image_size=image_size,
        save_slices=args.save_slices
    )
    print(f"\n✅ Preprocessing complete!")
