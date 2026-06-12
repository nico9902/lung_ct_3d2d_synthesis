import os
import numpy as np
import nibabel as nib


def extract_annotation_center(slice_annotation):
    if np.any(slice_annotation > 0):
        y, x = np.where(slice_annotation > 0)

        center_x = np.mean(x)
        center_y = np.mean(y)

        radius_x = (np.max(x) - np.min(x)) / 2.0
        radius_y = (np.max(y) - np.min(y)) / 2.0

        return np.array([center_x, center_y, radius_x, radius_y], dtype=np.float32)

    else:
        return np.array([-1, -1, -1, -1], dtype=np.float32)


if __name__ == "__main__":
    dir_path = "/ssd2/domenico/datasets/LIDC-IDRI_nifti"
    radius_list = []
    for patient in sorted(os.listdir(dir_path)):
        patient_path = os.path.join(dir_path, patient)
        
        if os.path.isdir(patient_path):
            for file in os.listdir(patient_path):
                if file.endswith("nodule_mask.nii.gz"):
                    file_path = os.path.join(patient_path, file)
                    annotation = nib.load(file_path).get_fdata()
                    annotation = annotation.transpose(2, 0, 1)
                    annotation = np.array(
                        [extract_annotation_center(s) for s in annotation]
                    )
                    valid = annotation[:, 2:] >= 0
                    valid = valid.all(axis=1)
                    radius_list.append(annotation[valid, 2:])

    if len(radius_list) > 0:
        radius_list = np.concatenate(radius_list, axis=0)
        mean_radius = np.mean(radius_list, axis=0)
        print(f"mean radius = {mean_radius}")
    else:
        print("No annotations found")