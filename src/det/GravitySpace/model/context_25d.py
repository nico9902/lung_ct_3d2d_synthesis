import numpy as np
import torch


def extract_25d_slices(volume, center_z: int, n_slices: int):
    """
    Extract an odd-sized neighboring-slice stack as ``N x H x W``.

    Border candidates are padded by replicating the nearest valid slice, so
    ``center_z=0, n_slices=3`` yields ``[0, 0, 1]``.
    Supports NumPy arrays and torch tensors with shape ``D x H x W``.
    """
    if n_slices <= 0 or n_slices % 2 == 0:
        raise ValueError(f"n_slices must be a positive odd integer, got {n_slices}.")
    if volume.ndim != 3:
        raise ValueError(f"volume must have shape D x H x W, got {tuple(volume.shape)}.")

    depth = volume.shape[0]
    if depth <= 0:
        raise ValueError("volume depth must be positive.")

    center_z = int(round(center_z))
    half = n_slices // 2
    indices = [min(max(z, 0), depth - 1) for z in range(center_z - half, center_z + half + 1)]

    if torch.is_tensor(volume):
        return torch.stack([volume[idx] for idx in indices], dim=0)
    if isinstance(volume, np.ndarray):
        return np.stack([volume[idx] for idx in indices], axis=0)
    raise TypeError(f"Unsupported volume type: {type(volume)!r}. Expected torch.Tensor or np.ndarray.")
