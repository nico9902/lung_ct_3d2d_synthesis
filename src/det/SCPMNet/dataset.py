from __future__ import annotations

from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import Dataset


def load_volume(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    volume = nib.load(str(path)).get_fdata().astype(np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume at {path}, got shape {volume.shape}.")
    # LIDC NIfTI files are loaded as x/y/z, while SCPM-Net and the target
    # builder use depth/height/width = z/y/x.
    return volume.transpose(2, 1, 0)


def normalize_ct(volume: np.ndarray, clip: tuple[float, float]) -> np.ndarray:
    volume = np.clip(volume, clip[0], clip[1])
    return ((volume - clip[0]) / (clip[1] - clip[0]) * 2.0 - 1.0).astype(np.float32)


def pad_crop(volume: np.ndarray, start: np.ndarray, crop_size: np.ndarray, pad_value: float = -1.0) -> tuple[np.ndarray, np.ndarray]:
    end = start + crop_size
    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.asarray(volume.shape))
    dst_start = src_start - start
    out = np.full(crop_size.tolist(), pad_value, dtype=np.float32)
    out[
        dst_start[0] : dst_start[0] + src_end[0] - src_start[0],
        dst_start[1] : dst_start[1] + src_end[1] - src_start[1],
        dst_start[2] : dst_start[2] + src_end[2] - src_start[2],
    ] = volume[src_start[0] : src_end[0], src_start[1] : src_end[1], src_start[2] : src_end[2]]
    return out, src_start


def resolve_image_path(row: pd.Series, data_root: Path) -> Path:
    seriesuid = str(row["seriesuid"]) if "seriesuid" in row and pd.notna(row["seriesuid"]) else ""
    raw_path = str(row["image_path"]) if "image_path" in row and pd.notna(row["image_path"]) else ""
    candidates: list[Path] = []
    if raw_path:
        path = Path(raw_path)
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(data_root / path)
            parts = path.parts
            if "lidc_process" in parts:
                idx = parts.index("lidc_process")
                candidates.append(data_root / Path(*parts[idx + 1 :]))
            if seriesuid and seriesuid in parts:
                idx = parts.index(seriesuid)
                candidates.append(data_root / Path(*parts[idx:]))
    if seriesuid:
        candidates.append(data_root / seriesuid / f"{seriesuid}_volume.nii.gz")
        candidates.append(data_root / seriesuid / f"{seriesuid}.nii.gz")
        candidates.append(data_root / f"{seriesuid}_volume.nii.gz")
        candidates.append(data_root / f"{seriesuid}.nii.gz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else Path(raw_path)


def prepare_label_dataframe(csv_path: str, split: str, data_root: Path, skip_missing_images: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"].astype(str) == split].copy()
    if "seriesuid" not in df.columns:
        df["seriesuid"] = df["image_path"].apply(lambda p: Path(str(p)).stem)
    if df.empty:
        return df
    df["_resolved_image_path"] = df.apply(lambda row: str(resolve_image_path(row, data_root)), axis=1)
    if skip_missing_images:
        df = df[df["_resolved_image_path"].map(lambda p: Path(p).exists())].copy()
    return df


class SCPMCSVVolumeDataset(Dataset):
    """3D CT dataset for SCPM-Net.

    Expected CSV columns: `seriesuid`, `split`, `image_path`, and either
    `coordZ, coordY, coordX, radius`, `coordZ, coordY, coordX, diameter`,
    or LIDC-style `x, y, z, w, h, d/depth`.
    Coordinates are converted to voxel-space z/y/x after loading the volume.
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        data_root: str = "",
        crop_size: tuple[int, int, int] = (96, 96, 96),
        samples_per_volume: int = 1,
        clip: tuple[float, float] = (-1000.0, 400.0),
        positive_crop_prob: float = 0.7,
        mask_path_column: str | None = None,
        skip_missing_images: bool = True,
        deterministic_seed: int | None = None,
    ):
        self.csv_path = csv_path
        self.split = split
        self.data_root = Path(data_root) if data_root else Path("")
        self.crop_size = np.asarray(crop_size, dtype=np.int64)
        self.samples_per_volume = max(1, int(samples_per_volume))
        self.clip = clip
        self.positive_crop_prob = positive_crop_prob
        self.mask_path_column = mask_path_column
        self.deterministic_seed = deterministic_seed
        df = prepare_label_dataframe(csv_path, split, self.data_root, skip_missing_images)
        self.groups = [(str(k), g.reset_index(drop=True)) for k, g in df.groupby("seriesuid", sort=False)]

    def __len__(self) -> int:
        return len(self.groups) * self.samples_per_volume

    def _resolve(self, path: str) -> Path:
        path = Path(str(path))
        return path if path.is_absolute() else self.data_root / path

    def _annotations_from_rows(self, rows: pd.DataFrame, shape: tuple[int, int, int]) -> np.ndarray:
        if self.mask_path_column and self.mask_path_column in rows.columns and pd.notna(rows.iloc[0][self.mask_path_column]):
            mask = load_volume(self._resolve(rows.iloc[0][self.mask_path_column])) > 0
            labeled, count = ndimage.label(mask)
            boxes = []
            for label_id in range(1, count + 1):
                pts = np.argwhere(labeled == label_id)
                if pts.size == 0:
                    continue
                center = pts.mean(axis=0)
                radius = max(np.linalg.norm(pts - center, axis=1).max(), 1.0)
                boxes.append([center[0], center[1], center[2], radius])
            return np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

        if all(col in rows.columns for col in ["coordZ", "coordY", "coordX"]):
            coord_cols = ("coordZ", "coordY", "coordX")
        elif all(col in rows.columns for col in ["z", "y", "x"]):
            coord_cols = ("z", "y", "x")
        else:
            return np.zeros((0, 4), dtype=np.float32)

        valid = rows.dropna(subset=list(coord_cols))
        if "label" in valid.columns:
            valid = valid[valid["label"].astype(str).str.lower().isin(("nodule", "1", "true", "positive"))]
        boxes = []
        for _, row in valid.iterrows():
            radius = self._row_radius(row)
            if radius and radius > 0:
                boxes.append([float(row[coord_cols[0]]), float(row[coord_cols[1]]), float(row[coord_cols[2]]), float(radius)])
        return np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

    @staticmethod
    def _row_radius(row: pd.Series) -> float:
        if "radius" in row and pd.notna(row["radius"]):
            return float(row["radius"])
        for diameter_col in ("diameter", "diameter_mm"):
            if diameter_col in row and pd.notna(row[diameter_col]):
                return float(row[diameter_col]) / 2.0
        depth_col = "depth" if "depth" in row and pd.notna(row["depth"]) else "d"
        dims = []
        for col in ("w", "h", depth_col):
            if col in row and pd.notna(row[col]):
                dims.append(float(row[col]))
        if len(dims) == 3:
            return max(dims) / 2.0
        return 0.0

    def _rng(self, index: int):
        if self.deterministic_seed is None:
            return np.random
        return np.random.RandomState(int(self.deterministic_seed) + int(index))

    def _crop_start(self, shape: np.ndarray, annotations: np.ndarray, rng=np.random) -> np.ndarray:
        max_start = np.maximum(shape - self.crop_size, 0)
        if len(annotations) and rng.rand() < self.positive_crop_prob:
            center = annotations[rng.randint(len(annotations)), :3]
            jitter = rng.uniform(-0.25, 0.25, size=3) * self.crop_size
            start = np.floor(center - self.crop_size / 2 + jitter).astype(np.int64)
        else:
            start = np.asarray([rng.randint(v + 1) if v > 0 else 0 for v in max_start], dtype=np.int64)
        return np.minimum(np.maximum(start, 0), max_start)

    def _crop_annotations(self, annotations: np.ndarray, src_start: np.ndarray) -> np.ndarray:
        crop_end = src_start + self.crop_size
        if len(annotations):
            centers = annotations[:, :3]
            keep = np.all((centers >= src_start) & (centers < crop_end), axis=1)
            annotations = annotations[keep].copy()
            annotations[:, :3] -= src_start
        return annotations

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        seriesuid, rows = self.groups[index // self.samples_per_volume]
        #print(f"Loading seriesuid={seriesuid} for sample index {index} with {len(rows)} annotation rows.")
        image_path = Path(rows.iloc[0]["_resolved_image_path"])
        volume = normalize_ct(load_volume(image_path), self.clip)
        annotations = self._annotations_from_rows(rows, volume.shape)
        start = self._crop_start(np.asarray(volume.shape), annotations, self._rng(index))
        crop, src_start = pad_crop(volume, start, self.crop_size)
        annotations = self._crop_annotations(annotations, src_start)
        return {
            "image": torch.from_numpy(crop[None]),
            "annot": torch.from_numpy(annotations.astype(np.float32)),
            "origin": torch.from_numpy(src_start.astype(np.float32)),
            "seriesuid": seriesuid,
        }


def sliding_starts(shape: np.ndarray, crop_size: np.ndarray, stride: np.ndarray) -> list[np.ndarray]:
    starts_per_axis = []
    for dim, crop, step in zip(shape.tolist(), crop_size.tolist(), stride.tolist()):
        max_start = max(dim - crop, 0)
        starts = list(range(0, max_start + 1, max(int(step), 1)))
        if starts[-1] != max_start:
            starts.append(max_start)
        starts_per_axis.append(starts)
    return [np.asarray(start, dtype=np.int64) for start in product(*starts_per_axis)]


class SCPMSlidingWindowDataset(Dataset):
    """Deterministic full-volume sliding-window dataset for test/inference."""

    def __init__(
        self,
        csv_path: str,
        split: str,
        data_root: str = "",
        crop_size: tuple[int, int, int] = (96, 96, 96),
        stride: tuple[int, int, int] = (24, 24, 24),
        clip: tuple[float, float] = (-1000.0, 400.0),
        mask_path_column: str | None = None,
        skip_missing_images: bool = True,
        include_annotations: bool = False,
    ):
        self.csv_path = csv_path
        self.split = split
        self.data_root = Path(data_root) if data_root else Path("")
        self.crop_size = np.asarray(crop_size, dtype=np.int64)
        self.stride = np.asarray(stride, dtype=np.int64)
        self.clip = clip
        self.mask_path_column = mask_path_column
        self.include_annotations = include_annotations
        self._cached_path: Path | None = None
        self._cached_volume: np.ndarray | None = None
        self._annotation_helper = None
        if include_annotations:
            self._annotation_helper = SCPMCSVVolumeDataset(
                csv_path=csv_path,
                split=split,
                data_root=data_root,
                crop_size=crop_size,
                clip=clip,
                mask_path_column=mask_path_column,
                skip_missing_images=skip_missing_images,
            )
        df = prepare_label_dataframe(csv_path, split, self.data_root, skip_missing_images)
        self.groups = [(str(k), g.reset_index(drop=True)) for k, g in df.groupby("seriesuid", sort=False)]
        self.windows: list[tuple[str, Path, pd.DataFrame, np.ndarray, np.ndarray]] = []
        for seriesuid, rows in self.groups:
            image_path = Path(rows.iloc[0]["_resolved_image_path"])
            shape = np.asarray(load_volume(image_path).shape, dtype=np.int64)
            for start in sliding_starts(shape, self.crop_size, self.stride):
                self.windows.append((seriesuid, image_path, rows, start, shape))

    def __len__(self) -> int:
        return len(self.windows)

    def _resolve(self, path: str) -> Path:
        path = Path(str(path))
        return path if path.is_absolute() else self.data_root / path

    def _load_cached_volume(self, image_path: Path) -> np.ndarray:
        if self._cached_path == image_path and self._cached_volume is not None:
            return self._cached_volume
        self._cached_path = image_path
        self._cached_volume = normalize_ct(load_volume(image_path), self.clip)
        return self._cached_volume

    def _annotations_from_rows(self, rows: pd.DataFrame, shape: tuple[int, int, int]) -> np.ndarray:
        if self._annotation_helper is None:
            return np.zeros((0, 4), dtype=np.float32)
        return self._annotation_helper._annotations_from_rows(rows, shape)

    def _crop_annotations(self, annotations: np.ndarray, src_start: np.ndarray) -> np.ndarray:
        crop_end = src_start + self.crop_size
        if len(annotations):
            centers = annotations[:, :3]
            keep = np.all((centers >= src_start) & (centers < crop_end), axis=1)
            annotations = annotations[keep].copy()
            annotations[:, :3] -= src_start
        return annotations

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        seriesuid, image_path, rows, start, shape = self.windows[index]
        volume = self._load_cached_volume(image_path)
        crop, src_start = pad_crop(volume, start, self.crop_size)
        sample = {
            "image": torch.from_numpy(crop[None]),
            "origin": torch.from_numpy(src_start.astype(np.float32)),
            "volume_shape": torch.from_numpy(shape.astype(np.float32)),
            "seriesuid": seriesuid,
        }
        if self.include_annotations:
            annotations = self._annotations_from_rows(rows, tuple(volume.shape))
            annotations = self._crop_annotations(annotations, src_start)
            sample["annot"] = torch.from_numpy(annotations.astype(np.float32))
        return sample


def scpm_collate(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    images = torch.stack([item["image"] for item in batch])
    max_ann = max(int(item["annot"].shape[0]) for item in batch)
    annots = torch.zeros((len(batch), max(max_ann, 1), 4), dtype=torch.float32)
    for i, item in enumerate(batch):
        annot = item["annot"]
        if annot.numel():
            annots[i, : annot.shape[0]] = annot
    output = {"image": images, "annot": annots, "seriesuid": [str(item["seriesuid"]) for item in batch]}
    if all("origin" in item for item in batch):
        output["origin"] = torch.stack([item["origin"] for item in batch])
    return output


def scpm_sliding_collate(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "origin": torch.stack([item["origin"] for item in batch]),
        "volume_shape": torch.stack([item["volume_shape"] for item in batch]),
        "seriesuid": [str(item["seriesuid"]) for item in batch],
    }
