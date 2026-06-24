import os
import sys
import torch
import numpy as np
import pydicom
from PIL import Image
from scipy.interpolate import Rbf
from itertools import product
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path so we can import modules from src
sys.path.append(os.getcwd())

import src.builder as builder
import src.datamodule as datamodule

from scipy.ndimage import label as nd_label
from scipy.ndimage import center_of_mass

def remove_overlapping_nodules(labeled_mask, num_features):
    """
    Remove smaller nodules when they overlap with larger ones.
    
    Args:
        labeled_mask: 3D array where each nodule has a unique label (1, 2, 3, ...)
        num_features: Number of nodules found
    
    Returns:
        cleaned_mask: Labeled mask with only non-overlapping nodules (smaller ones removed)
        remaining_labels: List of labels that were kept
    """
    if num_features == 0:
        return labeled_mask, []
    
    # Calculate volume for each nodule
    nodule_volumes = {}
    for label_id in range(1, num_features + 1):
        volume = np.sum(labeled_mask == label_id)
        nodule_volumes[label_id] = volume
    
    # Sort nodules by volume (largest first)
    sorted_nodules = sorted(nodule_volumes.items(), key=lambda x: x[1], reverse=True)
    
    # Track which nodules to keep
    kept_labels = []
    cleaned_mask = np.zeros_like(labeled_mask)
    
    for label_id, volume in sorted_nodules:
        current_nodule_mask_3d = (labeled_mask == label_id)
        current_nodule_mask_xy = np.any(current_nodule_mask_3d, axis=0)
        
        # Check if this nodule overlaps with any already kept nodule
        overlaps = False
        for kept_label in kept_labels:
            kept_nodule_mask_3d = (cleaned_mask == kept_label)
            kept_nodule_mask_xy = np.any(kept_nodule_mask_3d, axis=0)
            # Check for intersection
            intersection = np.logical_and(current_nodule_mask_xy, kept_nodule_mask_xy)
            if np.any(intersection):
                overlaps = True
                print(f"  Nodule {label_id} (volume={volume}) overlaps with nodule {kept_label} (volume={nodule_volumes[kept_label]}). Removing smaller nodule {label_id}.")
                break
        
        if not overlaps:
            # Keep this nodule
            cleaned_mask[current_nodule_mask_3d] = label_id
            kept_labels.append(label_id)
    
    print(f"  Kept {len(kept_labels)} out of {num_features} nodules after removing overlaps.")
    return cleaned_mask, kept_labels

def save_surfaces(save_path, dataset, cfg):
    for classe, tipo in product(["normal", "malignant"], ["surfaces"]):
        if not os.path.exists(os.path.join(save_path, classe, tipo)):
            os.makedirs(os.path.join(save_path, classe, tipo))

    # Default parameters
    # h, w = 512, 512 
    
    # Try to get parameters from config if available
    # try:
    #     # Assuming first transform is Resize or similar with 'size' attribute
    #     h, w = cfg.data.train_transforms.transforms[0].size
    # except Exception as e:
    #     print(f"Warning: Could not extract h, w or points from cfg: {e}. Using defaults: {h}x{w}, {num_points} points.")

    for k in range(len(dataset)):
        try:
            sample = dataset[k]
            if len(sample) == 4:
                img, label, patient_id, mask = sample
            else:
                img, label, patient_id = sample
                mask = None

            print(f"Processing patient: {patient_id}")

            # get width and height from img
            h, w = img.shape[2:]
            print(h, w)

            if label.item() == 0:
                save_path_images = os.path.join(save_path, "normal")
            else:
                save_path_images = os.path.join(save_path, "malignant")

            # Create subdirectories for surfaces
            current_save_dir = os.path.join(save_path_images, "surfaces", patient_id)
            os.makedirs(current_save_dir, exist_ok=True)

            save_file_path = os.path.join(current_save_dir, f"surface_{patient_id}.png")

            if os.path.exists(save_file_path):
                continue

            matrix = None
            if mask is not None: # and label.item() == 1:
                # Malignant case with mask: Use centroids of each separate nodule
                mask_np = (mask.numpy() > 0).astype(int)
                
                # Label connected components (nodules)
                labeled_mask, num_features = nd_label(mask_np)
                
                if num_features > 0:
                    # Remove overlapping nodules (keep only the larger ones)
                    print(f"Found {num_features} nodules for patient {patient_id}")
                    labeled_mask, kept_labels = remove_overlapping_nodules(labeled_mask, num_features)
                    for label_id in kept_labels:
                        coords = np.where(labeled_mask == label_id)
                    
                    # Calculate center of mass for each remaining labeled feature
                    matrix_list = []
                    for label_id in kept_labels:
                        # 1. Isola la maschera del singolo nodulo
                        nodule_mask = (labeled_mask == label_id)
                        coords = np.where(nodule_mask)
                        
                        # 2. Trova la slice Z con l'area maggiore (più rappresentativa)
                        z_coords = coords[0]
                        z_values, counts = np.unique(z_coords, return_counts=True)
                        best_z = z_values[np.argmax(counts)]
                        
                        # 3. Campiona punti da QUELLA slice specifica
                        # Invece di un solo punto, ne prendiamo diversi per "appiattire" lo spline sul nodulo
                        y_coords_at_z = coords[1][z_coords == best_z]
                        x_coords_at_z = coords[2][z_coords == best_z]
                        
                        # Prendiamo i 4 estremi + il centro per quel nodulo nella slice best_z
                        points_to_add = [
                            [best_z, np.mean(y_coords_at_z), np.mean(x_coords_at_z)], # Centro
                            [best_z, np.min(y_coords_at_z), np.min(x_coords_at_z)],   # Angoli locali
                            [best_z, np.max(y_coords_at_z), np.max(x_coords_at_z)],
                            [best_z, np.min(y_coords_at_z), np.max(x_coords_at_z)],
                            [best_z, np.max(y_coords_at_z), np.min(x_coords_at_z)]
                        ]
                        matrix_list.extend(points_to_add)

                    matrix = np.array(matrix_list)
                    print(f"Using {len(matrix)} nodule centers from mask.")
                else:
                    print(f"Warning: Malignant label but no nodules found in mask for {patient_id}.")
            
            if matrix is None:
                # Fallback for Normal cases or if mask is missing/empty: mid-slice flat surface
                mid_z = img.shape[0] // 2
                print(f"No mask or normal case: creating flat surface for {patient_id} at z={mid_z}")
                # Use 4 corners of the mid-slice as interpolation points
                matrix = np.array([
                    [mid_z, 0, 0],
                    [mid_z, 0, w - 1],
                    [mid_z, h - 1, 0],
                    [mid_z, h - 1, w - 1]
                ])

            # 1. Calculate the target depth based on the nodules found
            if len(matrix) > 0:
                target_z = np.mean(matrix[:, 0]) # Use the average Z of the nodules
            else:
                target_z = img.shape[0] // 2

            # 2. Create a robust boundary frame (Anchors)
            # We add points at corners AND mid-edges to "tack down" the surface
            edge_coords = [0, w // 2, w - 1]
            boundary_points = []
            for x_edge in edge_coords:
                for y_edge in [0, h - 1]:
                    boundary_points.append([target_z, y_edge, x_edge])
            for y_edge in [h // 2]: # Add mid-points of vertical edges
                for x_edge in [0, w - 1]:
                    boundary_points.append([target_z, y_edge, x_edge])

            matrix = np.vstack([matrix, np.array(boundary_points)])

            # 3. Apply RBF with Smoothing
            # function='thin_plate' does not use epsilon, but it DOES use 'smooth'
            rbf_spline = Rbf(
                matrix[:, 2], 
                matrix[:, 1], 
                matrix[:, 0], 
                function='thin_plate', 
                smooth=0.1  # <--- Crucial: prevents wild oscillations
            )

            x_range = np.arange(w)
            y_range = np.arange(h)
            X, Y = np.meshgrid(x_range, y_range)
            Z_spline = rbf_spline(X, Y)
            
            # Ensure Z is within volume bounds
            max_z = img.shape[0] - 1
            Z_spline = np.clip(Z_spline, 0, max_z)
            Z_spline = np.round(Z_spline).astype(int)
            
            import matplotlib.pyplot as plt
            X, Y = np.meshgrid(
                np.arange(Z_spline.shape[1]),
                np.arange(Z_spline.shape[0])
            )

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot_surface(X, Y, Z_spline, cmap='viridis', alpha=0.7)
            ax.scatter(matrix[:, 2], matrix[:, 1], matrix[:, 0], c='red', s=15)

            plt.show()
            # plt.imshow(Z_spline)
            # plt.scatter(matrix[:, 2], matrix[:, 1], c='red', s=10)
            # plt.show()
            # input("Premi INVIO per chiudere")

            output_image = np.zeros((h, w))
            img_np = img.numpy() # Shape: (D, 1, H, W)

            # Slice the volume according to the interpolated surface
            for i in range(h):
                for j in range(w):
                    z = Z_spline[i, j]
                    # Assuming img_np is (D, 1, H, W)
                    output_image[i, j] = img_np[z, 0, i, j]

            # Save result as image
            Image.fromarray((output_image * 255).astype(np.uint8)).convert("RGB").save(save_file_path)
        except Exception as e:
            print(f"Error processing item {k}: {e}")
            continue

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Ensure return_mask is True for this task
    OmegaConf.set_struct(cfg, False)
    cfg.data.return_mask = True
    
    print("Building datasets...")
    # Instantiate DataModule to get transforms
    dm = datamodule.DataModule(cfg.data)
    
    train_dataset = builder.build_dataset(cfg.data, split='train', transforms=None)
    val_dataset = builder.build_dataset(cfg.data, split='val', transforms=None)
    test_dataset = builder.build_dataset(cfg.data, split='test', transforms=None)

    save_path = "Lung2Dsynt_gt_nodules/"
    print(f"Path where surfaces will be saved: {save_path}")

    if train_dataset:
        print("Processing training set...")
        save_surfaces(os.path.join(save_path, "training"), train_dataset, cfg)
    if val_dataset:
        print("Processing validation set...")
        save_surfaces(os.path.join(save_path, "validation"), val_dataset, cfg)
    if test_dataset:
        print("Processing test set...")
        save_surfaces(os.path.join(save_path, "test"), test_dataset, cfg)

if __name__ == "__main__":
    main()