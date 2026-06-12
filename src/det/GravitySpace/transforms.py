import torch
import numpy as np
import cv2
from scipy.ndimage import label
from scipy.spatial.distance import cdist
import sys

class Custom_Resize:
    """
    Optimized resize using OpenCV instead of PIL (5-10x faster).
    Handles both slices and annotations in a sample.
    """
    def __init__(self, size):
        self.size = tuple(size) if isinstance(size, list) else size

    def __call__(self, sample):
        slices = sample['slices']
        annotations = sample['annotations']
        slices_resized = []
        annotations_resized = []
        
        for i in range(slices.shape[0]):
            slice_i = slices[i]
            annotation_i = annotations[i]

            # torch → numpy
            if isinstance(slice_i, torch.Tensor):
                slice_i = slice_i.cpu().numpy()
            if isinstance(annotation_i, torch.Tensor):
                annotation_i = annotation_i.cpu().numpy()

            # Resize slice using OpenCV (INTER_AREA is best for downsampling)
            slice_i = cv2.resize(slice_i, self.size, interpolation=cv2.INTER_AREA)
            slices_resized.append(slice_i)

            # Resize annotation using OpenCV with NEAREST interpolation
            # Note: cv2.resize takes (width, height)
            annotation_i = cv2.resize(annotation_i, self.size, interpolation=cv2.INTER_NEAREST)
            
            # Ensure exact same spatial dimensions as slice
            if annotation_i.shape != slice_i.shape:
                h_slice, w_slice = slice_i.shape
                h_anno, w_anno = annotation_i.shape
                annotation_i = annotation_i[:h_slice, :w_slice]
            
            annotations_resized.append(annotation_i)

        sample['slices'] = np.stack(slices_resized, axis=0)
        sample['annotations'] = np.stack(annotations_resized, axis=0)
        
        # Sanity check: ensure number of slices match
        assert sample['slices'].shape[0] == sample['annotations'].shape[0], \
            f"Slices and annotations must have same number of slices: {sample['slices'].shape[0]} vs {sample['annotations'].shape[0]}"

        return sample

class CustomToTensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        slices = sample['slices']
        # Se è già un tensore, converte solo in float
        if isinstance(slices, torch.Tensor):
            sample['slices'] = slices.float()
            return sample
        
        # Se è numpy, converte in tensore float
        sample['slices'] = torch.from_numpy(slices).float()
        
        return sample

class RepeatChannels:
    def __init__(self, repeats=3):
        self.repeats = repeats

    def __call__(self, sample):
        """
        Repeats channels along the second dimension (C).
        volume: [S, C, H, W]
        """
        slices = sample['slices']
        if isinstance(slices, np.ndarray):
            # If [S, H, W], add channel dim
            if slices.ndim == 3:
                slices = np.expand_dims(slices, 1)
            sample['slices'] = np.repeat(slices, self.repeats, axis=1)
            return sample
        elif isinstance(slices, torch.Tensor):
            # If [S, H, W], add channel dim
            if slices.dim() == 3:
                slices = slices.unsqueeze(1)
            
            # Repeat along C dimension (dim=1)
            # volume shape is [S, C, H, W]
            sizes = [1] * slices.dim()
            sizes[1] = self.repeats
            sample['slices'] = slices.repeat(*sizes)
            return sample
        else:
            str_err = f"Unsupported type for slices: {type(slices)}"
            sys.exit(str_err)


class AdjacentSliceChannels:
    def __init__(self):
        self.context_slices = 3
        self.radius = 1

    def __call__(self, sample):
        """
        Builds 2.5D inputs by stacking slice i-1, slice i, and slice i+1 as channels.
        Input [S, H, W] becomes [S, 3, H, W].
        Border slices use edge replication.
        """
        slices = sample['slices']
        is_numpy = isinstance(slices, np.ndarray)

        if isinstance(slices, torch.Tensor):
            if slices.dim() == 4:
                if slices.shape[1] != 1:
                    raise ValueError(f"AdjacentSliceChannels expects single-channel input, got {tuple(slices.shape)}")
                slices = slices[:, 0]
            elif slices.dim() != 3:
                raise ValueError(f"AdjacentSliceChannels expects [S,H,W] or [S,1,H,W], got {tuple(slices.shape)}")
        elif is_numpy:
            if slices.ndim == 4:
                if slices.shape[1] != 1:
                    raise ValueError(f"AdjacentSliceChannels expects single-channel input, got {slices.shape}")
                slices = slices[:, 0]
            elif slices.ndim != 3:
                raise ValueError(f"AdjacentSliceChannels expects [S,H,W] or [S,1,H,W], got {slices.shape}")
        else:
            str_err = f"Unsupported type for slices: {type(slices)}"
            sys.exit(str_err)

        channels = []
        num_slices = slices.shape[0]
        for offset in range(-self.radius, self.radius + 1):
            indices = np.clip(np.arange(num_slices) + offset, 0, num_slices - 1)
            if isinstance(slices, torch.Tensor):
                index_tensor = torch.as_tensor(indices, device=slices.device, dtype=torch.long)
                channels.append(slices.index_select(0, index_tensor))
            else:
                channels.append(slices[indices])

        if isinstance(slices, torch.Tensor):
            sample['slices'] = torch.stack(channels, dim=1)
        else:
            sample['slices'] = np.stack(channels, axis=1)
        return sample

# class ExtractAnnotationCenter:
#     def __init__(self, max_nodules=10):
#         self.max_nodules = max_nodules

#     def __call__(self, sample):
#         annotation = sample['annotations']
        
#         def extract_annotation_centers(slice_annotation):
#             # Init empty centers with -1 padding
#             centers = np.full((self.max_nodules, 4), -1, dtype=np.int32)
            
#             if np.any(slice_annotation > 0):
#                 # Find connected components (separate nodules)
#                 labeled_array, num_features = label(slice_annotation > 0)
                
#                 # Extract center and radius for each feature found
#                 for i in range(1, min(num_features + 1, self.max_nodules + 1)):
#                     y, x = np.where(labeled_array == i)
#                     center_x = np.mean(x)
#                     center_y = np.mean(y)

#                     radius_x = (np.max(x) - np.min(x)) // 2.0
#                     radius_y = (np.max(y) - np.min(y)) // 2.0

#                     centers[i-1] = [center_x, center_y, radius_x, radius_y]

#             return centers

#         sample['annotations'] = np.array([extract_annotation_centers(slice_annotation) for slice_annotation in annotation], dtype=np.int32)
#         return sample

class ExtractAnnotationCenter:
    def __init__(self, max_nodules=10):
        self.max_nodules = max_nodules

    def __call__(self, sample):
        annotation = sample['annotations']
        
        # Check if annotations are already precomputed
        if isinstance(annotation, dict) and annotation.get("__precomputed__"):
            # Use precomputed centers directly
            sample['annotations'] = annotation["centers"]
            return sample
        
        # Otherwise, extract from 3D mask (slow path)
        # Connected Components 3D
        labeled, num = label(annotation)

        components = [
            np.argwhere(labeled == i)
            for i in range(1, num + 1)
        ]

        # MERGE COMPONENTS (DISTANCE-BASED)
        def min_dist(a, b):
            return np.min(cdist(a, b))

        def merge_components(components, thresh=2.0):
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

        components = merge_components(components, thresh=2.0)
        
        def extract_from_slice(components, z):

            centers = np.full((self.max_nodules, 4), -1, dtype=np.float32)

            slice_components = []

            # =========================
            # 1. seleziona componenti attivi nella slice
            # =========================
            for comp in components:
                if np.any(comp[:, 0] == z):
                    slice_components.append(comp)

            # =========================
            # 2. estrai centri per ogni componente
            # =========================
            for i, comp in enumerate(slice_components[:self.max_nodules]):

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


        # =========================
        # APPLY PER SLICE
        # =========================
        sample['annotations'] = np.array([
            extract_from_slice(components, z)
            for z in range(annotation.shape[0])
        ], dtype=np.float32)

        return sample
