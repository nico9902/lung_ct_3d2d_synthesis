import os
import numpy as np
# Workaround for pylidc compatibility with newer numpy/python versions
try:
    np.int = int
    np.float = float
except:
    pass
import configparser
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

import SimpleITK as sitk
import pydicom
from tqdm import tqdm
import argparse
import json
from collections import defaultdict
import torch
import torch.nn.functional as F
import pylidc as pl
import pylidc.utils

def load_scan(path):
    """
    Loads a DICOM series from a folder.
    Returns a SimpleITK image.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def resample_z_only(itk_image, desired_z_spacing: float = 1.0, interpolator = sitk.sitkLinear, default_value = None):
    """
    Mantiene spacing e dimensione x,y invariati → cambia solo z
    Utile quando si vuole isotropia "leggera" senza alterare il piano assiale
    """
    original_spacing = itk_image.GetSpacing()
    original_size   = itk_image.GetSize()
    original_origin = itk_image.GetOrigin()
    direction       = itk_image.GetDirection()

    # Output spacing: x e y invariati, z = valore desiderato
    out_spacing = [original_spacing[0], original_spacing[1], desired_z_spacing]

    # Nuova dimensione solo lungo z
    scale_z = original_spacing[2] / desired_z_spacing
    out_size = [
        original_size[0],           # x invariato
        original_size[1],           # y invariato
        int(round(original_size[2] * scale_z))   # z adattato
    ]

    # Per mantenere allineamento con origine originale
    # (spesso è meglio, specialmente in radiologia)
    out_origin = list(original_origin)
    # Se vuoi centrare meglio lungo z puoi modificare qui:
    # out_origin[2] += (original_size[2] - out_size[2]) * original_spacing[2] * 0.5

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(direction)
    resampler.SetOutputOrigin(out_origin)
    resampler.SetInterpolator(interpolator)
    
    if default_value is not None:
        resampler.SetDefaultPixelValue(default_value)
    else:
        # Valore tipico per CT → -1000, per MR → 0, ...
        resampler.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    return resampler.Execute(itk_image)

def resample_isotropic(itk_image, out_spacing=[1.0, 1.0, 1.0], interpolator=sitk.sitkLinear):
    """
    Resamples an ITK image to isotropic spacing.
    """
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    resample.SetInterpolator(interpolator)

    return resample.Execute(itk_image)

def resize_volume(volume_np, target_shape=(128, 224, 224), mode='trilinear', align_corners=False):
    """
    Resizes a numpy volume (D, H, W) to target_shape using torch interpolate.
    """
    # Convert to torch tensor and add batch/channel dims: (1, 1, D, H, W)
    batch = torch.from_numpy(volume_np).unsqueeze(0).unsqueeze(0).float()
    
    # Resample
    if mode in ['nearest', 'area', 'nearest-exact']:
        resized = F.interpolate(batch, size=target_shape, mode=mode)
    else:
        resized = F.interpolate(batch, size=target_shape, mode=mode, align_corners=align_corners)
    
    return resized.squeeze().numpy()

def window_image(image, window_center, window_width):
    """
    Applies windowing to CT image.
    Common lung window: center=-600, width=1500 -> min=-1350, max=150
    LIDC-IDRI often uses: min=-1000, max=400
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(image, img_min, img_max)
    return windowed

def normalize(volume):
    """
    Normalize volume to [0, 1].
    Assumes windowing has already been applied.
    """
    v_min = volume.min()
    v_max = volume.max()
    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    return volume

def generate_3d_mask(patient_id, itk_img, consensus_threshold=1):
    """
    Generates a 3D nodule mask aligned with the loaded SimpleITK image.
    """
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
    if not scans:
        return None

    # Find the scan matching the volume depth
    scan = None
    volume_depth = itk_img.GetSize()[2]
    for s in scans:
        if len(s.slice_zvals) == volume_depth:
            scan = s
            break
    
    if scan is None:
        scan = scans[0]
        
    nodules = scan.cluster_annotations()
    
    # SimpleITK GetArrayFromImage returns (Z, Y, X)
    # We create mask in this orientation for direct alignment with numpy array
    full_mask = np.zeros(sitk.GetArrayFromImage(itk_img).shape, dtype=np.uint8)
    
    # Mapping of Z-coordinate to index
    # sitk image Origin and Spacing
    z_origin = itk_img.GetOrigin()[2]
    z_spacing = itk_img.GetSpacing()[2]
    
    for nodule_cluster in nodules:
        cmask, cbbox, _ = pl.utils.consensus(nodule_cluster, clevel=consensus_threshold)
        # cbbox is (slice(y_min, y_max), slice(x_min, x_max), slice(z_min, z_max))
        y_slice, x_slice, z_slice = cbbox
        
        # In pylidc, the consensus mask 'cmask' is (H, W, D) where Z is the last index
        # We need to map Z indices from the scan to our itk_img indices
        for z_idx_in_cmask in range(z_slice.stop - z_slice.start):
            z_idx_in_scan = z_slice.start + z_idx_in_cmask
            z_val = scan.slice_zvals[z_idx_in_scan]
            
            # Find closest Z-index in itk_img
            vol_idx = int(round((z_val - z_origin) / z_spacing))
            
            if 0 <= vol_idx < full_mask.shape[0]:
                # Update mask at vol_idx
                # full_mask is (Z, Y, X), so full_mask[vol_idx] is (Y, X)
                # cmask[:, :, z_idx_in_cmask] is (Y, X)
                full_mask[vol_idx][y_slice, x_slice] = np.logical_or(
                    full_mask[vol_idx][y_slice, x_slice], 
                    cmask[:, :, z_idx_in_cmask]
                )
    
    # Convert numpy mask back to sitk image for resampling
    mask_itk = sitk.GetImageFromArray(full_mask)
    mask_itk.SetOrigin(itk_img.GetOrigin())
    mask_itk.SetSpacing(itk_img.GetSpacing())
    mask_itk.SetDirection(itk_img.GetDirection())
    
    return mask_itk

def process_patient(patient_id, patient_dir, output_root, target_shape=(128, 224, 224), resize_shape=None, use_resample_z_only=False):
    """
    Complete pipeline for one patient including nodule masks.
    """
    try:
        # 1. Load scan
        itk_img = load_scan(patient_dir)
        
        # 2. Try to generate nodule mask
        itk_mask = generate_3d_mask(patient_id, itk_img)
        
        # 3. Isotropic resampling to 1.0mm
        if use_resample_z_only:
            itk_resampled = resample_z_only(itk_img)
            if itk_mask is not None:
                mask_resampled = resample_z_only(itk_mask)
        else:
            itk_resampled = resample_isotropic(itk_img, out_spacing=[1.0, 1.0, 1.0], interpolator=sitk.sitkLinear)
            if itk_mask is not None:
                mask_resampled = resample_isotropic(itk_mask, out_spacing=[1.0, 1.0, 1.0], interpolator=sitk.sitkNearestNeighbor)
        
        # 4. Convert to numpy
        volume = sitk.GetArrayFromImage(itk_resampled)
        if itk_mask is not None:
            mask = sitk.GetArrayFromImage(mask_resampled)
        
        # 5. Windowing & Normalize Volume
        volume = np.clip(volume, -1000, 400)
        volume = (volume + 1000) / 1400.0
        
        # 6. Resize to fixed dimensions
        if resize_shape is not None:
            volume_resized = resize_volume(volume, target_shape=resize_shape, mode='trilinear')
            if itk_mask is not None:
                mask_resized = resize_volume(mask.astype(np.float32), target_shape=resize_shape, mode='nearest')
                mask_resized = (mask_resized > 0.5).astype(np.uint8)
        else:
            volume_resized = volume
            if itk_mask is not None:
                mask_resized = mask
        
        # 7. Save
        patient_save_dir = os.path.join(output_root, patient_id)
        os.makedirs(patient_save_dir, exist_ok=True)
        
        np.save(os.path.join(patient_save_dir, "volume.npy"), volume_resized.astype(np.float32))
        if itk_mask is not None:
            np.save(os.path.join(patient_save_dir, "nodule_mask.npy"), mask_resized)
        
        # Save metadata
        metadata = {
            "patient_id": patient_id,
            "original_size": itk_img.GetSize(),
            "original_spacing": itk_img.GetSpacing(),
            "resampled_size": itk_resampled.GetSize(),
            "target_shape": target_shape,
            "has_mask": itk_mask is not None
        }
        with open(os.path.join(patient_save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        return True
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Preprocess LIDC-IDRI dataset with isotropic resampling and resizing.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw LIDC-IDRI data.")
    parser.add_argument("--output_dir", type=str, default="data/preprocessed", help="Path to save processed volumes.")
    parser.add_argument("--target_d", type=int, default=256, help="Target number of slices (depth).")
    parser.add_argument("--target_h", type=int, default=256, help="Target height.")
    parser.add_argument("--target_w", type=int, default=256, help="Target width.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of patients.")
    parser.add_argument("--resize_shape", type=int, nargs=3, default=None, help="Resize shape (D, H, W).")
    parser.add_argument("--resample_z_only", action="store_true", help="Resample only Z dimension.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    target_shape = (args.target_d, args.target_h, args.target_w)

    # Find all patient directories
    patients = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("LIDC-IDRI")]
    patients.sort()

    if args.limit:
        patients = patients[:args.limit]

    success_count = 0
    for p_id in tqdm(patients, desc="Processing Patients"):
        p_path = os.path.join(args.data_dir, p_id)
        
        # Find all directories containing DICOM files
        dicom_dirs = []
        for root, dirs, files in os.walk(p_path):
            if any(f.lower().endswith(".dcm") for f in files):
                dicom_dirs.append(root)
        
        if not dicom_dirs:
            print(f"No DICOM files found for {p_id}")
            continue
            
        # Select the directory with the most DICOM files (usually the full CT scan)
        series_dir = max(dicom_dirs, key=lambda d: len([f for f in os.listdir(d) if f.lower().endswith(".dcm")]))

        if process_patient(p_id, series_dir, args.output_dir, target_shape=target_shape, resize_shape=args.resize_shape, use_resample_z_only=args.resample_z_only):
            success_count += 1

    print(f"Successfully processed {success_count}/{len(patients)} patients.")

if __name__ == "__main__":
    main()
