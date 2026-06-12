import torch
import numpy as np
from torchvision.transforms import Resize, Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
import random
import cv2

class MinMaxNormalize:
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def __call__(self, volume):
        # numpy o torch tensor
        volume = Normalize(self.min,1)(volume)
        volume = Normalize(0,self.max-self.min)(volume)
        return volume

class Custom_Resize:
    """
    Optimized resize using OpenCV instead of PIL (5-10x faster).
    Supports both (H, W) and (C, H, W) slices.
    """
    def __init__(self, size):
        self.size = tuple(size) if isinstance(size, list) else size

    def __call__(self, volume):
        """
        Resize a volume to the target size.
        volume: [S, H, W] or [S, C, H, W] as numpy array or tensor
        returns: resized volume with same shape but new H, W dimensions
        """
        # Convert tensor to numpy if needed
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().numpy()
        
        S = volume.shape[0]
        resized_slices = []
        
        # Determine input shape format
        if volume.ndim == 3:  # [S, H, W]
            for i in range(S):
                slice_i = volume[i]  # [H, W]
                resized = cv2.resize(slice_i, self.size, interpolation=cv2.INTER_AREA)
                resized_slices.append(resized)
        elif volume.ndim == 4:  # [S, C, H, W]
            for i in range(S):
                slice_i = volume[i]  # [C, H, W]
                C = slice_i.shape[0]
                resized_channels = []
                for c in range(C):
                    channel = slice_i[c]  # [H, W]
                    resized = cv2.resize(channel, self.size, interpolation=cv2.INTER_AREA)
                    resized_channels.append(resized)
                resized_slice = np.stack(resized_channels, axis=0)  # [C, H, W]
                resized_slices.append(resized_slice)
        
        volume = np.stack(resized_slices, axis=0)
        return volume

class CustomToTensor:
    def __init__(self):
        pass

    def __call__(self, volume):
        # Se è già un tensore, converte solo in float
        if isinstance(volume, torch.Tensor):
            return volume.float()
        
        # Se è numpy, converte in tensore float
        volume = torch.from_numpy(volume).float()
        
        # Aggiungi il canale (1) se manca
        if volume.dim() == 3:  # [S, H, W]
            volume = volume.unsqueeze(1)  # [S, 1, H, W]
        
        return volume

class PartialVolumeTransform:
    """
    Applies a transformation to a random subset of slices in a volume.
    """
    def __init__(self, transform, p_slices=0.5):
        self.transform = transform
        self.p_slices = p_slices

    def __call__(self, volume):
        # volume: (D, H, W) or (D, C, H, W)
        is_numpy = isinstance(volume, np.ndarray)
        if is_numpy:
            # Keep as numpy for now if we want to use PIL-based transforms or just convert to tensor
            volume_tensor = torch.from_numpy(volume).float()
        else:
            volume_tensor = volume.clone()

        num_slices = volume_tensor.shape[0]
        num_to_augment = int(num_slices * self.p_slices)
        
        if num_to_augment == 0:
            return volume

        indices = torch.randperm(num_slices)[:num_to_augment].tolist()
        
        for i in indices:
            slice_i = volume_tensor[i]
            
            # If slice is (H, W), add channel dim (1, H, W) for torchvision
            if slice_i.ndim == 2:
                slice_i = slice_i.unsqueeze(0)
                transformed = self.transform(slice_i)
                volume_tensor[i] = transformed.squeeze(0)
            else:
                volume_tensor[i] = self.transform(slice_i)

        if is_numpy:
            return volume_tensor.numpy()
        return volume_tensor


class ConsistentVolumeTransform:
    """
    Applies the SAME transformation (same random parameters) to ALL slices in a volume.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, volume):
        is_numpy = isinstance(volume, np.ndarray)
        if is_numpy:
            volume_tensor = torch.from_numpy(volume).float()
        else:
            volume_tensor = volume.clone()

        # Set seed to ensure same transform parameters for all slices
        seed = np.random.randint(2147483647)
        
        outputs = []
        for i in range(volume_tensor.shape[0]):
            random.seed(seed)
            torch.manual_seed(seed)
            
            slice_i = volume_tensor[i]
            if slice_i.ndim == 2:
                slice_i = slice_i.unsqueeze(0)
                transformed = self.transform(slice_i)
                outputs.append(transformed.squeeze(0))
            else:
                outputs.append(self.transform(slice_i))

        volume_tensor = torch.stack(outputs, dim=0)
        
        if is_numpy:
            return volume_tensor.numpy()
        return volume_tensor

class RepeatChannels:
    def __init__(self, repeats=3):
        self.repeats = repeats

    def __call__(self, volume):
        """
        Repeats channels along the second dimension (C).
        volume: [S, C, H, W]
        """
        if isinstance(volume, np.ndarray):
            # If [S, H, W], add channel dim
            if volume.ndim == 3:
                volume = np.expand_dims(volume, 1)
            return np.repeat(volume, self.repeats, axis=1)
        elif isinstance(volume, torch.Tensor):
            # If [S, H, W], add channel dim
            if volume.dim() == 3:
                volume = volume.unsqueeze(1)
            
            # Repeat along C dimension (dim=1)
            # volume shape is [S, C, H, W]
            sizes = [1] * volume.dim()
            sizes[1] = self.repeats
            return volume.repeat(*sizes)
        return volume

class MIP:
    """
    Applies Maximum Intensity Projection (MIP) to a volume.
    """
    def __init__(self, slab_size=3, stride=1):
        self.slab_size = slab_size
        self.stride = stride

    def __call__(self, volume):
        """
        volume: [S, H, W] or [S, C, H, W]
        """
        is_numpy = isinstance(volume, np.ndarray)
        if is_numpy:
            volume_tensor = torch.from_numpy(volume).float()
        else:
            volume_tensor = volume.clone().float()

        # Ensure [S, C, H, W]
        if volume_tensor.ndim == 3:
            volume_tensor = volume_tensor.unsqueeze(1)
        
        S, C, H, W = volume_tensor.shape
        
        # Calculate output depth
        out_depth = (S - self.slab_size) // self.stride + 1
        
        if out_depth <= 0:
            # If volume is smaller than slab_size, return MIP of the whole volume
            mip_volume = torch.max(volume_tensor, dim=0, keepdim=True)[0]
            if is_numpy:
                return mip_volume.numpy()
            return mip_volume

        slabs = []
        for i in range(0, S - self.slab_size + 1, self.stride):
            slab = volume_tensor[i : i + self.slab_size]
            mip_slab = torch.max(slab, dim=0)[0]
            slabs.append(mip_slab)
            
        volume_mip = torch.stack(slabs, dim=0)

        if is_numpy:
            return volume_mip.numpy()
        return volume_mip