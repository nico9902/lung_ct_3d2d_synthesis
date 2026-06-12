import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from scipy.spatial.distance import cdist


class ExtractAnnotationCenter:
    def __init__(self, max_nodules=10):
        self.max_nodules = max_nodules

    def __call__(self, sample):
        annotation = sample["annotations"]

        if isinstance(annotation, dict) and annotation.get("__precomputed__"):
            sample["annotations"] = annotation["centers"]
            return sample

        labeled, num = label(annotation)

        components = [
            np.argwhere(labeled == i)
            for i in range(1, num + 1)
        ]

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
            # now: center_x, center_y, radius_x, radius_y, radius_z
            centers = np.full((self.max_nodules, 5), -1, dtype=np.float32)

            slice_components = []

            for comp in components:
                if np.any(comp[:, 0] == z):
                    slice_components.append(comp)

            for i, comp in enumerate(slice_components[:self.max_nodules]):

                coords = comp[comp[:, 0] == z]

                if len(coords) == 0:
                    continue

                y = coords[:, 1]
                x = coords[:, 2]

                all_z = comp[:, 0]
                all_y = comp[:, 1]
                all_x = comp[:, 2]

                center_x = np.mean(x)
                center_y = np.mean(y)

                radius_x = (np.max(all_x) - np.min(all_x)) / 2.0
                radius_y = (np.max(all_y) - np.min(all_y)) / 2.0
                radius_z = (np.max(all_z) - np.min(all_z)) / 2.0

                centers[i] = [
                    center_x,
                    center_y,
                    radius_x,
                    radius_y,
                    radius_z
                ]

            return centers

        sample["annotations"] = np.array([
            extract_from_slice(components, z)
            for z in range(annotation.shape[0])
        ], dtype=np.float32)

        return sample
    
if __name__ == "__main__":
    dir_path = "/ssd2/domenico/datasets/LIDC-IDRI_nifti_z_only"

    extractor = ExtractAnnotationCenter(max_nodules=10)
    radius_list = []

    for patient in sorted(os.listdir(dir_path)):
        patient_path = os.path.join(dir_path, patient)

        if os.path.isdir(patient_path):
            for file in os.listdir(patient_path):
                if file.endswith("nodule_mask.nii.gz"):
                    file_path = os.path.join(patient_path, file)

                    annotation = nib.load(file_path).get_fdata()
                    annotation = annotation.transpose(2, 0, 1)

                    sample = {"annotations": annotation}
                    sample = extractor(sample)

                    annotations = sample["annotations"]

                    # shape: [Z, max_nodules, 5]
                    radii = annotations[..., 2:5]

                    valid = np.all(radii >= 0, axis=-1)

                    radius_list.append(radii[valid])

    if len(radius_list) > 0:
        radius_list = np.concatenate(radius_list, axis=0)

        mean_radius_x, mean_radius_y, mean_radius_z = np.mean(radius_list, axis=0)

        print(f"mean radius x = {mean_radius_x}")
        print(f"mean radius y = {mean_radius_y}")
        print(f"mean radius z = {mean_radius_z}")
        print(f"mean radius [x, y, z] = {np.mean(radius_list, axis=0)}")
    else:
        print("No annotations found")