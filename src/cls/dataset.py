import torch
from collections import defaultdict
from PIL import Image
import os
import pydicom
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class LIDCRIDRIDataset2D(Dataset):
    def __init__(self, csv_file, processed_dir, transform=None, split=None):
        self.data_frame = pd.read_csv(csv_file)
        self.processed_dir = processed_dir
        self.split = split
        if split:
            self.data_frame = self.data_frame[self.data_frame['split'] == split].reset_index(drop=True)
            if self.split == 'train':
                self.processed_dir = os.path.join(processed_dir, 'training')
            elif self.split == 'val':
                self.processed_dir = os.path.join(processed_dir, 'validation')
            elif self.split == 'test':
                self.processed_dir = os.path.join(processed_dir, 'test')
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # get the row
        row = self.data_frame.iloc[idx]
        # get the patient id
        patient_id = row['patient_id']
        # get the label
        label = int(row['target'])
        # get the label name
        if label == 1:
            label_name = 'malignant'
        else:
            label_name = 'normal'
        # get the image path
        image_path = os.path.join(
            self.processed_dir,
            label_name,
            "surfaces",
            patient_id,
            f"surface_{patient_id}.png"
        )
        # load the image
        image = Image.open(image_path)

        # apply transform if provided
        if self.transform:
            image = self.transform(image)

        # retrieve label
        label = torch.tensor(label, dtype=torch.float32)
        
        # ensure correct shape: (H, W) -> (1, H, W)
        if image.ndim == 2: # (H, W)
            image = image.unsqueeze(0) # (1, H, W)
        
        return image, label, patient_id

class SliceLIDCRIDRIDataset(Dataset):
    def __init__(self, csv_file, processed_dir, transform=None, split=None, return_mask=False):
        self.data_frame = pd.read_csv(csv_file)
        self.processed_dir = processed_dir
        if split:
            self.data_frame = self.data_frame[self.data_frame['split'] == split].reset_index(drop=True)
        self.transform = transform
        self.return_mask = return_mask

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        patient_id = row['slice_id'].split('_')[0]
        slice_id = row['slice_id'].split('_')[1]
        volume_path = os.path.join(self.processed_dir, patient_id, f"volume.npy")
        volume = np.load(volume_path).astype(np.float32)
        image = volume[int(slice_id)]
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).unsqueeze(0)

        # retrieve label
        label = torch.tensor(label, dtype=torch.float32)

        # Handle tensor conversion if transform didn't do it
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        # Ensure correct shape: (H, W) -> (1, H, W)
        if image.ndim == 2: # (H, W)
            image = image.unsqueeze(0) # (1, H, W)
        
        # Return mask (optional)
        if self.return_mask:
            mask_path = os.path.join(self.processed_dir, patient_id, "nodule_mask", f"mask_volume.npy")
            volume_mask = np.load(mask_path).astype(np.float32)
            mask = volume_mask[int(slice_id)]

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            return image, label, row['slice_id'], mask
        
        return image, label, row['slice_id']


class LIDCIDRIDataset(Dataset):
    def __init__(self, root_dir, processed_dir, csv_file, split=None, transform=None, dicom=False, return_mask=False):
        """
        Args:
            root_dir: path to LIDC-IDRI/ or processed npy folder
            processed_dir: path to processed npy folder
            csv_file (string): Path to the csv file with annotations.
            split (string, optional): 'train', 'val', or 'test'. If None, loads all.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            dicom (bool): Whether to load from DICOM or processed npy files.
        """
        self.root_dir = root_dir
        self.processed_dir = processed_dir
        self.data_frame = pd.read_csv(csv_file)
        self.dicom = dicom
        self.return_mask = return_mask
        
        if split:
            self.data_frame = self.data_frame[self.data_frame['split'] == split].reset_index(drop=True)
            
        self.transform = transform
        
        # Each folder = one patient
        self.patients = self.data_frame['patient_id'].unique()
        self.labels = self.data_frame['target'].values.astype(int)

    def __len__(self):
        return len(self.patients)

    def _get_ct_series(self, patient_dir):
        """
        Groups CT slices by SeriesInstanceUID
        Assumes: patient_dir / study / series / *.dcm
        """
        series = defaultdict(list)

        for study_name in os.listdir(patient_dir):
            study_path = os.path.join(patient_dir, study_name)
            if not os.path.isdir(study_path):
                continue

            for series_name in os.listdir(study_path):
                series_path = os.path.join(study_path, series_name)
                if not os.path.isdir(series_path):
                    continue

                for fname in os.listdir(series_path):
                    # again: some datasets have no extension
                    if not fname.lower().endswith(".dcm"):
                        continue

                    path = os.path.join(series_path, fname)

                    try:
                        dcm = pydicom.dcmread(
                            path,
                            stop_before_pixels=True,
                            force=True,
                        )
                    except Exception as e:
                        print(f"Failed to read {path}: {e}")
                        continue

                    if getattr(dcm, "Modality", None) != "CT":
                        continue

                    if not hasattr(dcm, "ImagePositionPatient"):
                        continue

                    series_uid = getattr(dcm, "SeriesInstanceUID", None)
                    if series_uid is None:
                        continue

                    series[series_uid].append(path)

        if not series:
            raise RuntimeError(f"No CT series found in {patient_dir}")

        return series

    def _load_ct_volume(self, path, dicom=False):
        """
        Loads and sorts CT slices, converts to HU if DICOM
        """
        slices = []

        if dicom:
            for path_ in path:
                dcm = pydicom.dcmread(path_)
                img = dcm.pixel_array.astype(np.float32)

                # Convert to Hounsfield Units
                slope = float(getattr(dcm, "RescaleSlope", 1.0))
                intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
                img = img * slope + intercept

                z = float(dcm.ImagePositionPatient[2])
                slices.append((z, img))

            # sort by z-axis
            slices.sort(key=lambda x: x[0])
            volume = np.stack([s[1] for s in slices], axis=0)

            # windowing
            HU_MIN, HU_MAX = -1000, 400
            volume = np.clip(volume, HU_MIN, HU_MAX)  # LUNG window

            # min-max normalize
            volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)  # range [0,1]
        else:
            # path is a .npy file
            volume = np.load(path).astype(np.float32)

        return volume

    def __getitem__(self, idx):
        patient_id = self.patients[idx]

        if self.dicom:
            patient_dir = os.path.join(self.root_dir, patient_id)
            series_dict = self._get_ct_series(patient_dir)
            # select the largest CT series (true volume)
            path = max(series_dict.values(), key=len)
        else:
            # Load .npy slices
            patient_dir = os.path.join(self.processed_dir, patient_id)
            path = os.path.join(patient_dir, "volume.npy")
            if not path:
                raise RuntimeError(f"No .npy volume found in {os.path.join(self.processed_dir, patient_id)}")

        volume = self._load_ct_volume(path, dicom=self.dicom)

        # transform
        if self.transform:
            volume = self.transform(volume)
        
        # retrieve label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Ensure correct shape: (S, C, H, W) -> (D, 1, H, W)
        if volume.ndim == 3: # (D, H, W)
            volume = volume.unsqueeze(1) # (D, 1, H, W)

        if self.return_mask:
            if self.dicom:
                mask_path = os.path.join("data/processed_dicom", patient_id, "nodule_mask_3d.npy")
            else:
                mask_path = os.path.join("data/processed", patient_id, "nodule_mask", "mask_volume.npy")
            
            if os.path.exists(mask_path):
                mask = np.load(mask_path).astype(np.uint8)
                mask = torch.from_numpy(mask).float() # (D, H_orig, W_orig)
                
                # Resize mask to match volume spatial dimensions if they differ
                if mask.shape[1:] != volume.shape[2:]:
                    # F.interpolate expects (B, C, H, W)
                    mask = mask.unsqueeze(1) # (D, 1, H_orig, W_orig)
                    mask = torch.nn.functional.interpolate(mask, size=volume.shape[2:], mode='nearest')
                    mask = mask.squeeze(1) # (D, H_target, W_target)
                
                mask = mask.long()
            else:
                # If mask doesn't exist, return zeros of same shape as volume (depth, H, W)
                mask = torch.zeros(volume.shape[0], volume.shape[2], volume.shape[3]).long()
            return volume, label.unsqueeze(0), patient_id, mask

        return volume, label.unsqueeze(0), patient_id

# usage example
if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import transforms as transform
    from torchvision import transforms
    import sys
    import os

    # add project root to path so we can import modules from src
    sys.path.append(os.getcwd())

    transform = transforms.Compose([transform.ToTensorFloat32(), transform.Custom_Resize(size=[352, 480])])

    dataset = LIDCRIDRIDataset2D(
        processed_dir="data/Lung2Dsynt_gt_nodules",
        csv_file="data/batch_test_250.csv",
        transform=transform,
        split="train"
    )
    sample = dataset[0]
    print(f"Output shape: {sample[0].size()}")
    print(sample[0])

    # ct_volume is likely (D, 1, 256, 256) due to Dataset unsqueeze logic on 3D tensors
    # Wait, ToTensor returns (D, H, W). Resize returns (D, 256, 256).
    # Dataset logic: if ndim==3 -> unsqueeze(1) -> (D, 1, 256, 256).
    # Step 283 used [100][0] which implies (D, 1, H, W) or (1, D, H, W)?
    # Let's assume (D, 1, H, W) based on code.

    plt.imshow(sample[0][100][0], cmap="gray")
    plt.title(f"Patient ID: {sample[2]}")
    plt.axis("off")
    plt.show()

    print(ct_volume.min(), ct_volume.max())
    print(ct_volume.dtype)