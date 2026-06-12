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
import os
import pylidc as pl
import pylidc.utils
from tqdm import tqdm
import argparse
import json
import pydicom
from collections import defaultdict

def get_dicom_volume(patient_dir):
    """
    Loads DICOM slices from a directory and stacks them into a 3D volume.
    Returns: volume (D, H, W), z_coords, dicom_paths
    """
    series = defaultdict(list)
    for root, _, files in os.walk(patient_dir):
        for f in files:
            if not f.endswith(".dcm"): continue
            path = os.path.join(root, f)
            try:
                dcm = pydicom.dcmread(path, stop_before_pixels=True)
            except Exception: continue
            if getattr(dcm, "Modality", None) != "CT": continue
            if not hasattr(dcm, "ImagePositionPatient"): continue
            series[dcm.SeriesInstanceUID].append(path)

    if len(series) == 0:
        return None, None, None
        
    # Pick the series with the most slices (usually the main CT)
    dicom_paths = max(series.values(), key=len)
    
    slices = []
    for path in dicom_paths:
        dcm = pydicom.dcmread(path)
        z = float(dcm.ImagePositionPatient[2])
        img = dcm.pixel_array.astype(np.float32)
        # We don't necessarily need to rescale HU if we only care about the mask alignment,
        # but it's good practice.
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        img = img * slope + intercept
        slices.append((z, img))

    # Sort by Z-coordinate
    slices.sort(key=lambda x: x[0])
    volume = np.stack([s[1] for s in slices], axis=0)
    z_coords = [s[0] for s in slices]
    
    return volume, z_coords, dicom_paths

def generate_3d_mask(patient_id, volume_shape, z_coords, consensus_threshold=1):
    """
    Generates a 3D nodule mask aligned with the loaded DICOM volume.
    """
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
    if not scans:
        print(f"No scan found in pylidc for {patient_id}")
        return None

    # Find the scan matching the volume depth
    scan = None
    for s in scans:
        if s.to_volume().shape[2] == volume_shape[0]:
            scan = s
            break
    
    if scan is None:
        scan = scans[0]
        print(f"Warning: Depth mismatch for {patient_id}. Pylidc: {scan.to_volume().shape[2]}, Loaded: {volume_shape[0]}. Attempting alignment by Z-coordinates.")

    nodules = scan.cluster_annotations()
    full_mask = np.zeros(volume_shape, dtype=np.uint8)
    
    # scan.slice_zvals are sorted by z-coordinate by pylidc
    scan_z_coords = [float(z) for z in scan.slice_zvals]
    
    # Create a mapping from z-coordinate to index in our volume
    z_to_idx = {float(np.round(z, 4)): i for i, z in enumerate(z_coords)}
    
    for nodule_cluster in nodules:
        # pl.utils.consensus returns (mask, bbox, cluster)
        cmask, cbbox, _ = pl.utils.consensus(nodule_cluster, clevel=consensus_threshold)
        
        # cbbox is (slice(y_min, y_max), slice(x_min, x_max), slice(z_min, z_max))
        y_slice, x_slice, z_slice = cbbox
        for z_idx_in_scan in range(z_slice.start, z_slice.stop):
            z_val = float(np.round(scan_z_coords[z_idx_in_scan], 4))
            if z_val in z_to_idx:
                vol_idx = z_to_idx[z_val]
                # cmask is (H, W, D), so we need cmask[:, :, z_idx_in_scan - z_slice.start]
                # Note: cmask orientation in pylidc is (y, x, z) matching DICOM (row, col, slice)
                full_mask[vol_idx][y_slice, x_slice] = np.logical_or(
                    full_mask[vol_idx][y_slice, x_slice], 
                    cmask[:, :, z_idx_in_scan - z_slice.start]
                )
    
    return full_mask

def process_patient(patient_id, patient_dir, output_root, consensus_threshold=1):
    volume, z_coords, _ = get_dicom_volume(patient_dir)
    if volume is None:
        print(f"No DICOM data found for {patient_id}")
        return

    mask_3d = generate_3d_mask(patient_id, volume.shape, z_coords, consensus_threshold)
    if mask_3d is None:
        return

    patient_output_dir = os.path.join(output_root, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    mask_save_path = os.path.join(patient_output_dir, "nodule_mask_3d.npy")
    np.save(mask_save_path, mask_3d)
    
    # Optional: save volume as well if needed, or just the mask as requested.
    # np.save(os.path.join(patient_output_dir, "volume_3d.npy"), volume)

    print(f"Saved 3D nodule mask for {patient_id} to {mask_save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate 3D nodule masks from DICOM without preprocessing.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw LIDC-IDRI data.")
    parser.add_argument("--output_dir", type=str, default="data/processed_3d", help="Path to save 3D masks.")
    parser.add_argument("--consensus", type=int, default=1, help="Consensus threshold for nodules.")
    parser.add_argument("--limit", type=int, default=None, help="Limit patients.")
    args = parser.parse_args()

    patients = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("LIDC-IDRI")]
    patients.sort()

    if args.limit:
        patients = patients[:args.limit]

    for p_id in tqdm(patients, desc="Generating 3D masks"):
        p_dir = os.path.join(args.data_dir, p_id)
        process_patient(p_id, p_dir, args.output_dir, args.consensus)

if __name__ == "__main__":
    main()
