import os
import pydicom
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

from lungmask import LMInferer
from collections import defaultdict
from tqdm import tqdm
import argparse
from torchvision.transforms import Resize
from PIL import Image
import json
import pylidc as pl
import pylidc.utils

class LungPreprocessor:
    def __init__(self, model_name='R231', force_cpu=False):
        """
        Initializes the LMInferer.
        model_name: 'R231', 'LTRCLobes', 'R231CovidWeb'
        """
        self.inferer = LMInferer(modelname=model_name, force_cpu=force_cpu)

    def load_ct_volume(self, patient_dir):
        """
        Loads and sorts CT slices, converts to HU.
        Adapted from src/dataset.py
        Returns: (volume, z_coords)
        """
        series = defaultdict(list)
        for root, _, files in os.walk(patient_dir):
            for f in files:
                if not f.endswith(".dcm"):
                    continue
                path = os.path.join(root, f)
                try:
                    dcm = pydicom.dcmread(path, stop_before_pixels=True)
                except Exception:
                    continue
                if getattr(dcm, "Modality", None) != "CT":
                    continue
                if not hasattr(dcm, "ImagePositionPatient"):
                    continue
                series[dcm.SeriesInstanceUID].append(path)

        if len(series) == 0:
            return None, None

        # Select the largest CT series
        dicom_paths = max(series.values(), key=len)
        
        slices = []
        for path in dicom_paths:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            
            # Convert to Hounsfield Units
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
            img = img * slope + intercept
            
            z = float(dcm.ImagePositionPatient[2])
            slices.append((z, img))

        slices.sort(key=lambda x: x[0])
        volume = np.stack([s[1] for s in slices], axis=0)
        z_coords = [s[0] for s in slices]
        
        return volume, z_coords
    
    def compute_z_coords(self, volume):
        """
        Computes z-coordinates for the volume.
        """
        return np.arange(volume.shape[0])

    def get_lung_masks(self, volume):
        """
        Generates lung masks for the volume.
        """
        # LMInferer expects (D, H, W)
        segmentation = self.inferer.apply(volume)
        # Lung masks are typically labels 1 and 2 (left and right lung)
        lung_mask = (segmentation > 0).astype(np.uint8)
        return lung_mask

    def get_global_bbox(self, masks, padding=10):
        """
        Calculates the union bounding box across all masked slices.
        """
        all_coords = np.argwhere(masks)
        if all_coords.size == 0:
            return None
        
        # masks is (D, H, W). We want bbox for (H, W) across all D.
        # all_coords is (N, 3) where columns are (D, H, W)
        y_coords = all_coords[:, 1]
        x_coords = all_coords[:, 2]
        
        y_min, x_min = y_coords.min(), x_coords.min()
        y_max, x_max = y_coords.max() + 1, x_coords.max() + 1
        
        # Add padding
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        # Assuming all slices have same H, W
        h, w = masks.shape[1], masks.shape[2]
        y_max = min(h, y_max + padding)
        x_max = min(w, x_max + padding)
        
        return int(y_min), int(x_min), int(y_max), int(x_max)

    def crop_and_resize(self, image, bbox, target_size=(224, 224)):
        """
        Crops the image using the provided bbox and resizes it.
        bbox: (y_min, x_min, y_max, x_max)
        """
        y_min, x_min, y_max, x_max = bbox
        cropped_image = image[y_min:y_max, x_min:x_max]
        
        if target_size:
            pil_img = Image.fromarray(cropped_image)
            # resized_img = Resize(target_size)(pil_img)
            cropped_image = np.array(pil_img)
            
        return cropped_image

    def generate_nodule_mask(self, patient_id, volume_shape, z_coords, consensus_threshold=1):
        """
        Generates a 3D nodule mask aligned with the loaded volume.
        """
        scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
        if not scans:
            return None

        # Find the scan matching the volume depth
        scan = None
        for s in scans:
            if len(s.slice_zvals) == volume_shape[0]:
                scan = s
                break
        
        if scan is None:
            scan = scans[0]

        nodules = scan.cluster_annotations()
        full_mask = np.zeros(volume_shape, dtype=np.uint8)
        
        # Create a mapping from z-coordinate to index in our volume
        z_to_idx = {float(np.round(z, 4)): i for i, z in enumerate(z_coords)}
        
        for nodule_cluster in nodules:
            cmask, cbbox, _ = pl.utils.consensus(nodule_cluster, clevel=consensus_threshold)
            y_slice, x_slice, z_slice = cbbox
            
            for z_idx_in_cmask in range(z_slice.stop - z_slice.start):
                z_idx_in_scan = z_slice.start + z_idx_in_cmask
                z_val = float(np.round(scan.slice_zvals[z_idx_in_scan], 4))
                
                if z_val in z_to_idx:
                    vol_idx = z_to_idx[z_val]
                    full_mask[vol_idx][y_slice, x_slice] = np.logical_or(
                        full_mask[vol_idx][y_slice, x_slice], 
                        cmask[:, :, z_idx_in_cmask]
                    )
        
        return full_mask

    def process_patient(self, patient_id, patient_dir, output_root, load_dicom):
        """
        Processes a single patient: extracts lungs, finds global bbox, crops and saves single 3D volume + mask.
        """
        if load_dicom:
            volume, z_coords = self.load_ct_volume(patient_dir)
        else:
            volume_path = os.path.join(patient_dir, "volume.npy")
            if not os.path.exists(volume_path):
                print(f"Error: {volume_path} not found.")
                return
            volume = np.load(volume_path)
            z_coords = self.compute_z_coords(volume)

        if volume is None:
            print(f"No CT volume found for patient {patient_id}")
            return

        print(f"Volume min: {volume.min():.4f}, max: {volume.max():.4f}")

        # Try to load existing lung mask if not loading DICOM
        lung_masks = None
        mask_path = os.path.join(patient_dir, "lung_mask.npy")
        if not load_dicom and os.path.exists(mask_path):
            print(f"Loading existing lung masks for {patient_id}...")
            lung_masks = np.load(mask_path)
        
        if lung_masks is None:
            print(f"Generating lung masks for {patient_id}...")
            # If volume is normalized [0, 1], un-normalize to HU for LMInferer
            segmentation_volume = volume
            if volume.max() <= 1.0 and volume.min() >= 0.0:
                print("Un-normalizing volume for segmentation...")
                segmentation_volume = volume * 1400.0 - 1000.0
            
            lung_masks = self.get_lung_masks(segmentation_volume)

        bbox = self.get_global_bbox(lung_masks)
        if bbox is None:
            print(f"No lungs detected for patient {patient_id}")
            return
            
        # Handle nodule mask: generate if loading DICOM, otherwise try to load existing
        nodule_mask = None
        if load_dicom:
            print(f"Generating nodule masks for {patient_id} using pylidc...")
            nodule_mask = self.generate_nodule_mask(patient_id, volume.shape, z_coords)
        else:
            nodule_mask_path = os.path.join(patient_dir, "nodule_mask.npy")
            if os.path.exists(nodule_mask_path):
                print(f"Loading existing nodule mask for {patient_id}...")
                nodule_mask = np.load(nodule_mask_path)

        patient_output_dir = os.path.join(output_root, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Save the lung mask for future use
        np.save(os.path.join(patient_output_dir, "lung_mask.npy"), lung_masks)
        
        kept_volume_slices = []
        kept_nodule_slices = []
        pixel_values = []
        
        HU_MIN, HU_MAX = -1000, 400

        for i in tqdm(range(volume.shape[0]), desc=f"Processing {patient_id}", leave=False):
            mask_slice = lung_masks[i]
            if np.any(mask_slice):
                # Crop and process slice
                cropped_slice = self.crop_and_resize(volume[i], bbox)
                
                # Only normalize if it wasn't already normalized
                if volume.max() > 1.0:
                    cropped_slice = np.clip(cropped_slice, HU_MIN, HU_MAX)
                    cropped_slice = (cropped_slice - HU_MIN) / (HU_MAX - HU_MIN)
                
                kept_volume_slices.append(cropped_slice)
                pixel_values.extend(cropped_slice.flatten().tolist())

                if nodule_mask is not None:
                    # Nodule mask should be the same depth as volume
                    cropped_nodule_mask = self.crop_and_resize(nodule_mask[i], bbox)
                    kept_nodule_slices.append(cropped_nodule_mask)

        if not kept_volume_slices:
            print(f"No valid slices kept for patient {patient_id}")
            return

        # Stack into 3D arrays
        final_volume = np.stack(kept_volume_slices, axis=0)
        np.save(os.path.join(patient_output_dir, "volume.npy"), final_volume)

        if kept_nodule_slices:
            final_nodule_mask = np.stack(kept_nodule_slices, axis=0)
            np.save(os.path.join(patient_output_dir, "nodule_mask.npy"), final_nodule_mask)

        # Calculate stats
        pixel_values = np.array(pixel_values)
        stats = {
            "patient_id": patient_id,
            "bbox": {
                "y_min": bbox[0], "x_min": bbox[1], "y_max": bbox[2], "x_max": bbox[3],
                "height": bbox[2] - bbox[0], "width": bbox[3] - bbox[1]
            },
            "slices": {
                "total": volume.shape[0], "kept": len(kept_volume_slices)
            },
            "statistics": {
                "mean": float(np.mean(pixel_values)) if pixel_values.size > 0 else 0,
                "std": float(np.std(pixel_values)) if pixel_values.size > 0 else 0,
                "min": float(np.min(pixel_values)) if pixel_values.size > 0 else 0,
                "max": float(np.max(pixel_values)) if pixel_values.size > 0 else 0
            },
            "has_nodule_mask": nodule_mask is not None
        }
        
        with open(os.path.join(patient_output_dir, "metadata.json"), "w") as f:
            json.dump(stats, f, indent=4)
        
        print(f"Finished {patient_id}: saved volume ({final_volume.shape}). Nodule mask: {'Yes' if kept_nodule_slices else 'No'}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess CT scans: extract lungs and crop.")
    parser.add_argument("--data_dir", type=str, default="data/preprocessed", help="Path to raw LIDC-IDRI data.")
    parser.add_argument("--output_dir", type=str, default="data/preprocessed/processed", help="Path to save processed slices.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of patients to process.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage for LMInferer.")
    parser.add_argument("--load_dicom", action="store_true", help="Whether to load DICOM files (default: False, loads .npy)")
    args = parser.parse_args()

    preprocessor = LungPreprocessor(force_cpu=args.cpu)
    
    # Get patient list from data_dir
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist.")
        return

    patients = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("LIDC-IDRI")]
    patients.sort()
    
    if args.limit:
        patients = patients[:args.limit]
        
    for p_id in tqdm(patients, desc="Processing patients"):
        p_dir = os.path.join(args.data_dir, p_id)
        preprocessor.process_patient(p_id, p_dir, args.output_dir, args.load_dicom)

if __name__ == "__main__":
    main()