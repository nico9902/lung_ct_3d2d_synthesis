import numpy as np
np.int = int
np.float = float
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
from preprocess import LungPreprocessor

class NoduleMaskGenerator(LungPreprocessor):
    def __init__(self, consensus_threshold=1, force_cpu=False):
        """
        Initializes the mask generator.
        consensus_threshold: Number of radiologists that must agree for a voxel to be part of a nodule.
        """
        super().__init__(force_cpu=force_cpu)
        self.consensus_threshold = consensus_threshold

    def get_full_nodule_mask(self, scan):
        """
        Generates a 3D binary mask for all nodules in the scan.
        """
        # scan.to_volume() returns (H, W, D) by default in pylidc
        # but our preprocess.py uses (D, H, W) based on DICOM loading.
        # We need to be careful about orientation.
        
        # cluster_annotations returns a list of lists of Annotation objects
        nodules = scan.cluster_annotations()
        
        # Get volume shape from scan
        vol = scan.to_volume()
        mask = np.zeros(vol.shape, dtype=np.uint8)
        
        for nodule_cluster in nodules:
            # consensus_mask returns a (H, W, D) mask and a bbox
            cmask, cbbox = pl.utils.consensus(nodule_cluster, clevel=self.consensus_threshold)
            mask[cbbox] = np.logical_or(mask[cbbox], cmask)
            
        # Pylidc volume is (H, W, D). We usually want (D, H, W) to match preprocess.py
        # preprocess.py loads slices and stacks them: volume = np.stack([s[1] for s in slices], axis=0)
        # s[1] is pixel_array which is (H, W). So stack is (D, H, W).
        # Scan.to_volume() returns (H, W, D).
        # Let's transpose to (D, H, W)
        mask = mask.transpose(2, 0, 1)
        
        return mask

    def process_patient_masks(self, patient_id, output_root):
        """
        Processes a single patient: extracts nodules, crops and saves mask slices.
        """
        # Query scan using pylidc
        scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
        if not scans:
            print(f"No scan found in pylidc for patient {patient_id}")
            return

        # Usually there's only one main CT scan per patient ID in LIDC-IDRI, 
        # but preprocess.py picks the one with most slices.
        # Let's find the scan that matches the one used in preprocess.py if possible,
        # or just take the first one if only one exists.
        scan = scans[0] 
        # If there are multiple, we might need a better selection logic.
        # But for now, let's assume one main scan.
        
        # We need the lung mask to get the same BBox as preprocess.py
        # We'll load the volume used by preprocess.py to ensure identical mapping.
        # This is because pylidc might load slices in a different order or different series.
        
        # Actually, let's look at how preprocess.py finds the patient_dir.
        # It's better to pass the patient_dir.
        
        # We'll re-implement the part of process_patient to get the exact same bbox.
        pass

    def run_on_patient(self, patient_id, patient_dir, output_root):
        """
        Matches preprocess.py logic to save masks in the same coordinate space.
        """
        # Load volume and get z-coordinates (z, img)
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

        if len(series) == 0: return
        dicom_paths = max(series.values(), key=len)
        
        slices = []
        for path in dicom_paths:
            dcm = pydicom.dcmread(path)
            z = float(dcm.ImagePositionPatient[2])
            img = dcm.pixel_array.astype(np.float32)
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
            img = img * slope + intercept
            slices.append((z, img))

        slices.sort(key=lambda x: x[0])
        volume = np.stack([s[1] for s in slices], axis=0)
        z_coords = [s[0] for s in slices]

        # Get lung masks to find the same bbox as preprocess.py
        lung_masks = self.get_lung_masks(volume)
        bbox = self.get_global_bbox(lung_masks)
        if bbox is None:
            return

        # Get nodule mask from pylidc
        scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
        if not scans:
            print(f"No scan found in pylidc for {patient_id}")
            return
        
        # Find the scan matching the volume depth
        scan = None
        for s in scans:
            if s.to_volume().shape[2] == volume.shape[0]:
                scan = s
                break
        
        if scan is None:
            scan = scans[0]
            print(f"Warning: Depth mismatch for {patient_id}. Pylidc: {scan.to_volume().shape[2]}, Loaded: {volume.shape[0]}. Attempting alignment.")

        # pylidc.Scan.to_volume() returns volume in same order as scan.slices
        # Let's get the nodule mask and align it slice by slice
        nodules = scan.cluster_annotations()
        full_mask = np.zeros(volume.shape, dtype=np.uint8)
        
        # scan.slice_zvals are sorted by z-coordinate by pylidc
        scan_z_coords = [float(z) for z in scan.slice_zvals]
        
        # Create a mapping from z-coordinate to index in our volume
        z_to_idx = {float(z): i for i, z in enumerate(z_coords)}
        
        for nodule_cluster in nodules:
            # pl.utils.consensus returns (mask, bbox, cluster)
            cmask, cbbox, _ = pl.utils.consensus(nodule_cluster, clevel=self.consensus_threshold)
            
            # cbbox is (slice(y_min, y_max), slice(x_min, x_max), slice(z_min, z_max))
            z_slice = cbbox[2]
            for z_idx_in_scan in range(z_slice.start, z_slice.stop):
                z_val = scan_z_coords[z_idx_in_scan]
                if z_val in z_to_idx:
                    vol_idx = z_to_idx[z_val]
                    # cmask is (H, W, D), so we need cmask[:, :, z_idx_in_scan - z_slice.start]
                    full_mask[vol_idx][cbbox[0], cbbox[1]] = np.logical_or(
                        full_mask[vol_idx][cbbox[0], cbbox[1]], 
                        cmask[:, :, z_idx_in_scan - z_slice.start]
                    )

        patient_output_dir = os.path.join(output_root, patient_id)
        nodule_mask_dir = os.path.join(patient_output_dir, "nodule_mask")
        os.makedirs(nodule_mask_dir, exist_ok=True)

        nodules_metadata = []
        
        # scan.slice_zvals are sorted by z-coordinate by pylidc
        scan_z_coords = [float(z) for z in scan.slice_zvals]
        
        # Create a mapping from z-coordinate to index in our volume
        z_to_idx = {float(z): i for i, z in enumerate(z_coords)}
        
        for cluster_idx, nodule_cluster in enumerate(nodules):
            # pl.utils.consensus returns (mask, bbox, cluster)
            cmask, cbbox, _ = pl.utils.consensus(nodule_cluster, clevel=self.consensus_threshold)
            
            # cbbox is (slice(y_min, y_max), slice(x_min, x_max), slice(z_min, z_max))
            y_slice, x_slice, z_slice = cbbox
            
            # Original bbox coordinates
            orig_y_min, orig_y_max = y_slice.start, y_slice.stop
            orig_x_min, orig_x_max = x_slice.start, x_slice.stop
            # For Z, we use indices in our sorted volume
            # Find the range of indices in our volume that correspond to these z-values
            mapped_z_indices = []
            for z_idx_in_scan in range(z_slice.start, z_slice.stop):
                z_val = scan_z_coords[z_idx_in_scan]
                if z_val in z_to_idx:
                    mapped_z_indices.append(z_to_idx[z_val])
            
            if not mapped_z_indices:
                continue
                
            orig_z_min, orig_z_max = min(mapped_z_indices), max(mapped_z_indices) + 1

            # Processed bbox coordinates (relative to lung crop)
            # bbox is (y_min_lung, x_min_lung, y_max_lung, x_max_lung)
            l_y_min, l_x_min, l_y_max, l_x_max = bbox
            
            proc_y_min = orig_y_min - l_y_min
            proc_y_max = orig_y_max - l_y_min
            proc_x_min = orig_x_min - l_x_min
            proc_x_max = orig_x_max - l_x_min
            # Z stays same as its in volume space
            proc_z_min, proc_z_max = orig_z_min, orig_z_max

            nodules_metadata.append({
                "nodule_id": cluster_idx,
                "original_bbox": {
                    "y_min": int(orig_y_min), "y_max": int(orig_y_max),
                    "x_min": int(orig_x_min), "x_max": int(orig_x_max),
                    "z_min": int(orig_z_min), "z_max": int(orig_z_max)
                },
                "processed_bbox": {
                    "y_min": int(proc_y_min), "y_max": int(proc_y_max),
                    "x_min": int(proc_x_min), "x_max": int(proc_x_max),
                    "z_min": int(proc_z_min), "z_max": int(proc_z_max)
                }
            })

            for z_idx_in_scan in range(z_slice.start, z_slice.stop):
                z_val = scan_z_coords[z_idx_in_scan]
                if z_val in z_to_idx:
                    vol_idx = z_to_idx[z_val]
                    # cmask is (H, W, D), so we need cmask[:, :, z_idx_in_scan - z_slice.start]
                    full_mask[vol_idx][cbbox[0], cbbox[1]] = np.logical_or(
                        full_mask[vol_idx][cbbox[0], cbbox[1]], 
                        cmask[:, :, z_idx_in_scan - z_slice.start]
                    )

        # Save metadata JSON
        with open(os.path.join(nodule_mask_dir, "nodule_info.json"), "w") as f:
            json.dump(nodules_metadata, f, indent=4)

        for i in range(volume.shape[0]):
            mask_slice = lung_masks[i]
            if np.any(mask_slice):
                # Crop the nodule mask using the same bbox
                cropped_nodule_mask = self.crop_and_resize(full_mask[i], bbox, target_size=None)
                
                # Save as binary mask in the subfolder
                mask_save_path = os.path.join(nodule_mask_dir, f"nodule_mask_{i:04d}.npy")
                np.save(mask_save_path, (cropped_nodule_mask > 0).astype(np.uint8))

        print(f"Saved nodule masks and info for {patient_id}")

def main():
    parser = argparse.ArgumentParser(description="Generate nodule masks matching preprocessed slices.")
    parser.add_argument("--data_dir", type=str, default="/Users/domenicopaolo/Documents/datasets/LIDC-IDRI/manifest-1600709154662/LIDC-IDRI", help="Path to raw LIDC-IDRI data.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Path to save masks.")
    parser.add_argument("--consensus", type=int, default=1, help="Consensus threshold for nodules.")
    parser.add_argument("--limit", type=int, default=None, help="Limit patients.")
    args = parser.parse_args()

    generator = NoduleMaskGenerator(consensus_threshold=args.consensus)
    
    patients = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("LIDC-IDRI")]
    patients.sort()

    if args.limit:
        patients = patients[:args.limit]

    for p_id in tqdm(patients, desc="Generating nodule masks"):
        p_dir = os.path.join(args.data_dir, p_id)
        generator.run_on_patient(p_id, p_dir, args.output_dir)

if __name__ == "__main__":
    main()
