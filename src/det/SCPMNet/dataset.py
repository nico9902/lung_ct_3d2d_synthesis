from __future__ import annotations

from itertools import product
import os
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


def normalize_ct(volume: np.ndarray, clip: tuple[float, float], intensity_mode: str = "hu") -> np.ndarray:
    mode = str(intensity_mode).lower()
    if mode == "auto":
        finite = volume[np.isfinite(volume)]
        if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 255.0:
            mode = "uint8"
        else:
            mode = "hu"
    if mode == "uint8":
        volume = np.clip(volume, 0.0, 255.0)
        return (volume / 255.0 * 2.0 - 1.0).astype(np.float32)
    if mode != "hu":
        raise ValueError(f"Unsupported intensity_mode={intensity_mode!r}; expected 'hu', 'uint8', or 'auto'.")
    volume = np.clip(volume, clip[0], clip[1])
    return ((volume - clip[0]) / (clip[1] - clip[0]) * 2.0 - 1.0).astype(np.float32)


def normalized_volume_cache_path(
    cache_dir: str | Path,
    seriesuid: str,
    clip: tuple[float, float],
    intensity_mode: str,
) -> Path:
    safe_seriesuid = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in str(seriesuid))
    clip_tag = f"{float(clip[0]):g}_{float(clip[1]):g}".replace("-", "m").replace(".", "p")
    return Path(cache_dir) / f"{safe_seriesuid}_mode-{str(intensity_mode)}_clip-{clip_tag}.npy"


def load_normalized_volume(
    image_path: str | Path,
    clip: tuple[float, float],
    intensity_mode: str = "hu",
    normalized_volume_cache_dir: str | Path | None = None,
    seriesuid: str | None = None,
) -> np.ndarray:
    if not normalized_volume_cache_dir:
        return normalize_ct(load_volume(image_path), clip, intensity_mode)

    cache_dir = Path(normalized_volume_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = seriesuid or Path(image_path).name.replace("_volume.nii.gz", "").replace(".nii.gz", "").replace(".npy", "")
    cache_path = normalized_volume_cache_path(cache_dir, cache_key, clip, intensity_mode)
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")

    volume = normalize_ct(load_volume(image_path), clip, intensity_mode)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}.npy")
    try:
        np.save(tmp_path, volume)
        os.replace(tmp_path, cache_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return np.load(cache_path, mmap_mode="r")


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
        intensity_mode: str = "hu",
        normalized_volume_cache_dir: str | Path | None = None,
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
        self.intensity_mode = str(intensity_mode)
        self.normalized_volume_cache_dir = Path(normalized_volume_cache_dir) if normalized_volume_cache_dir else None
        self.positive_crop_prob = positive_crop_prob
        self.mask_path_column = mask_path_column
        self.deterministic_seed = deterministic_seed
        df = prepare_label_dataframe(csv_path, split, self.data_root, skip_missing_images)
        self.groups = [(str(k), g.reset_index(drop=True)) for k, g in df.groupby("seriesuid", sort=False)]

    def __len__(self) -> int:
        return len(self.groups)

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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[torch.Tensor] | str]:
        seriesuid, rows = self.groups[index]
        #print(f"Loading seriesuid={seriesuid} for sample index {index} with {len(rows)} annotation rows.")
        image_path = Path(rows.iloc[0]["_resolved_image_path"])
        volume = load_normalized_volume(
            image_path,
            self.clip,
            self.intensity_mode,
            self.normalized_volume_cache_dir,
            seriesuid=seriesuid,
        )
        annotations = self._annotations_from_rows(rows, volume.shape)

        crops = []
        crop_annotations = []
        origins = []
        for sample_idx in range(self.samples_per_volume):
            rng_index = index * self.samples_per_volume + sample_idx
            start = self._crop_start(np.asarray(volume.shape), annotations, self._rng(rng_index))
            crop, src_start = pad_crop(volume, start, self.crop_size)
            sample_annotations = self._crop_annotations(annotations, src_start)
            crops.append(torch.from_numpy(crop[None]))
            crop_annotations.append(torch.from_numpy(sample_annotations.astype(np.float32)))
            origins.append(torch.from_numpy(src_start.astype(np.float32)))

        return {
            "image": torch.stack(crops),
            "annot": crop_annotations,
            "origin": torch.stack(origins),
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
        intensity_mode: str = "hu",
        normalized_volume_cache_dir: str | Path | None = None,
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
        self.intensity_mode = str(intensity_mode)
        self.normalized_volume_cache_dir = Path(normalized_volume_cache_dir) if normalized_volume_cache_dir else None
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
                intensity_mode=intensity_mode,
                normalized_volume_cache_dir=normalized_volume_cache_dir,
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
        seriesuid = image_path.name.replace("_volume.nii.gz", "").replace(".nii.gz", "").replace(".npy", "")
        self._cached_volume = load_normalized_volume(
            image_path,
            self.clip,
            self.intensity_mode,
            self.normalized_volume_cache_dir,
            seriesuid=seriesuid,
        )
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


def scpm_collate(batch: list[dict[str, torch.Tensor | list[torch.Tensor] | str]]) -> dict[str, torch.Tensor | list[str]]:
    image_chunks = []
    annotation_chunks = []
    origin_chunks = []
    seriesuids = []

    for item in batch:
        image = item["image"]
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected tensor image, got {type(image)!r}.")
        if image.ndim == 4:
            image = image.unsqueeze(0)
        elif image.ndim != 5:
            raise ValueError(f"Expected image with 4 or 5 dims, got shape {tuple(image.shape)}.")
        image_chunks.append(image)

        item_annotations = item["annot"]
        if isinstance(item_annotations, torch.Tensor):
            if item_annotations.ndim == 2:
                annotation_chunks.append(item_annotations)
            elif item_annotations.ndim == 3:
                annotation_chunks.extend([annot for annot in item_annotations])
            else:
                raise ValueError(f"Expected annotations with 2 or 3 dims, got shape {tuple(item_annotations.shape)}.")
        else:
            annotation_chunks.extend(item_annotations)

        seriesuids.extend([str(item["seriesuid"])] * int(image.shape[0]))

        if "origin" in item:
            origin = item["origin"]
            if not isinstance(origin, torch.Tensor):
                raise TypeError(f"Expected tensor origin, got {type(origin)!r}.")
            if origin.ndim == 1:
                origin = origin.unsqueeze(0)
            origin_chunks.append(origin)

    images = torch.cat(image_chunks, dim=0)
    max_ann = max(int(annot.shape[0]) for annot in annotation_chunks)
    annots = torch.zeros((len(annotation_chunks), max(max_ann, 1), 4), dtype=torch.float32)
    for i, annot in enumerate(annotation_chunks):
        if annot.numel():
            annots[i, : annot.shape[0]] = annot

    output = {"image": images, "annot": annots, "seriesuid": seriesuids}
    if all("origin" in item for item in batch):
        output["origin"] = torch.cat(origin_chunks, dim=0)
    return output


def scpm_sliding_collate(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "origin": torch.stack([item["origin"] for item in batch]),
        "volume_shape": torch.stack([item["volume_shape"] for item in batch]),
        "seriesuid": [str(item["seriesuid"]) for item in batch],
    }
