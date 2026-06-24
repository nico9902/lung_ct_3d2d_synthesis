import os
import sys
import hashlib
import traceback
import torch
import numpy as np
try:
    import pydicom
except ImportError:
    pydicom = None
import pandas as pd
from PIL import Image
from pathlib import Path
from scipy.interpolate import Rbf
from itertools import product
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

# Add project root to path so we can import modules from src
sys.path.append(os.getcwd())

from scipy.ndimage import label as nd_label
from scipy.ndimage import binary_erosion, binary_fill_holes


def keep_largest_components(mask, max_components=2):
    mask = np.asarray(mask) > 0
    labeled_mask, num_features = nd_label(mask)
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)

    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0
    keep_labels = np.argsort(component_sizes)[-int(max_components):]
    keep_labels = keep_labels[component_sizes[keep_labels] > 0]
    return np.isin(labeled_mask, keep_labels)


def keep_largest_components_per_slice(mask, max_components=1, min_area=1):
    mask = np.asarray(mask) > 0
    cleaned = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        labeled_slice, num_features = nd_label(mask[z])
        if num_features == 0:
            continue
        component_sizes = np.bincount(labeled_slice.ravel())
        component_sizes[0] = 0
        keep_labels = np.argsort(component_sizes)[-int(max_components):]
        keep_labels = keep_labels[component_sizes[keep_labels] >= int(min_area)]
        if len(keep_labels) > 0:
            cleaned[z] = np.isin(labeled_slice, keep_labels)
    return cleaned


def estimate_body_mask(volume, body_threshold_percentile=1):
    volume = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(volume)
    if not np.any(finite):
        return np.zeros(volume.shape, dtype=bool)

    finite_values = volume[finite]
    looks_like_hu = np.nanmin(finite_values) < -100 and np.nanmax(finite_values) > 100
    if looks_like_hu:
        tissue_mask = volume > -600
    else:
        low = np.percentile(finite_values, float(np.clip(body_threshold_percentile, 0, 20)))
        high = np.percentile(finite_values, 99)
        threshold = low + 0.05 * max(high - low, 1e-6)
        tissue_mask = volume > threshold

    body_mask = np.zeros_like(tissue_mask, dtype=bool)
    for z in range(tissue_mask.shape[0]):
        filled = binary_fill_holes(tissue_mask[z])
        body_mask[z] = filled

    body_mask = keep_largest_components_per_slice(body_mask, max_components=1)
    if not np.any(body_mask):
        print("Warning: body mask estimation found no body voxels.")
    return body_mask


def get_lung_mask(
    volume,
    model_name="R231",
    force_cpu=False,
    method="body_threshold",
    normalized_air_threshold=0.35,
    hu_air_min=-1000,
    hu_air_max=-320,
    body_threshold_percentile=1,
    lung_component_count=2,
):
    """Return a deterministic body-constrained lung/air mask for [D, H, W]."""
    del model_name, force_cpu
    if method != "body_threshold":
        print(f"Warning: unsupported lung_mask_method={method}; using body_threshold.")

    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected CT volume shape [D, H, W], got {volume.shape}.")

    body_mask = estimate_body_mask(
        volume,
        body_threshold_percentile=body_threshold_percentile,
    )
    if not np.any(body_mask):
        return np.zeros(volume.shape, dtype=bool)

    finite_values = volume[np.isfinite(volume)]
    looks_like_hu = np.nanmin(finite_values) < -100 and np.nanmax(finite_values) > 100
    if looks_like_hu:
        air_mask = np.logical_and(volume >= hu_air_min, volume <= hu_air_max)
    else:
        air_mask = volume <= normalized_air_threshold

    lung_mask = np.logical_and(air_mask, body_mask)
    lung_mask = keep_largest_components_per_slice(
        lung_mask,
        max_components=max(1, int(lung_component_count)),
        min_area=20,
    )
    lung_mask = keep_largest_components(
        lung_mask,
        max_components=max(1, int(lung_component_count)),
    )
    if not np.any(lung_mask):
        print("Warning: body-constrained lung mask found no lung voxels.")
    return lung_mask


def stable_patient_seed(patient_id):
    digest = hashlib.sha256(str(patient_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def mid_slice_plane(depth, h, w):
    mid_z = depth // 2
    return np.array([
        [mid_z, 0, 0],
        [mid_z, 0, w - 1],
        [mid_z, h - 1, 0],
        [mid_z, h - 1, w - 1],
    ], dtype=float)


def sample_pseudo_regions_inside_lung(
    patient_id,
    lung_mask,
    min_regions=1,
    max_regions=3,
    min_radius=8,
    max_radius=20,
    erode_iterations=2,
    central_percentile=70,
    seed_offset=0,
):
    lung_mask = np.asarray(lung_mask) > 0
    if lung_mask.ndim != 3:
        raise ValueError(f"Expected lung_mask shape [D, H, W], got {lung_mask.shape}.")

    depth, h, w = lung_mask.shape
    lung_points = np.argwhere(lung_mask)
    if len(lung_points) == 0:
        print("Warning: no lung voxels found for pseudo-region sampling.")
        return np.empty((0, 3), dtype=float)

    lung_centroid = lung_points.mean(axis=0)
    print(f"Patient: {patient_id}")
    print(f"Lung centroid: {np.round(lung_centroid, 2).tolist()}")

    sampling_mask = lung_mask
    if erode_iterations > 0 and np.any(lung_mask):
        eroded = binary_erosion(lung_mask, iterations=int(erode_iterations))
        if np.any(eroded):
            sampling_mask = eroded
        else:
            print("Warning: eroded lung mask is empty; sampling from un-eroded lung mask.")

    candidate_points = np.argwhere(sampling_mask)
    if len(candidate_points) == 0:
        print("Warning: no lung voxels found for pseudo-region sampling.")
        return np.empty((0, 3), dtype=float)
    print(f"Candidate voxels: {len(candidate_points)}")

    central_percentile = float(np.clip(central_percentile, 1, 100))
    lung_distances = np.linalg.norm(lung_points - lung_centroid, axis=1)
    distance_threshold = np.percentile(lung_distances, central_percentile)
    candidate_distances = np.linalg.norm(candidate_points - lung_centroid, axis=1)
    central_points = candidate_points[candidate_distances <= distance_threshold]
    min_central_candidates = max(100, int(max_regions) * 9)
    print(f"Central candidate voxels: {len(central_points)}")
    if len(central_points) < min_central_candidates:
        print(
            f"Warning: central candidate subset too small "
            f"({len(central_points)} < {min_central_candidates}); sampling from full lung candidates."
        )
    else:
        candidate_points = central_points

    min_regions = max(1, int(min_regions))
    max_regions = max(min_regions, int(max_regions))
    min_radius = max(1, int(min_radius))
    max_radius = max(min_radius, int(max_radius))

    rng = np.random.default_rng(stable_patient_seed(patient_id) + int(seed_offset))
    num_regions = int(rng.integers(min_regions, max_regions + 1))
    selected_indices = rng.choice(len(candidate_points), size=min(num_regions, len(candidate_points)), replace=False)
    offsets = np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
        [0, -1, -1],
        [0, -1, 1],
        [0, 1, -1],
        [0, 1, 1],
    ], dtype=int)

    control_points = []
    for center in candidate_points[selected_indices]:
        radius = int(rng.integers(min_radius, max_radius + 1))
        points = center + offsets * radius
        points[:, 0] = np.clip(points[:, 0], 0, depth - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)
        points[:, 2] = np.clip(points[:, 2], 0, w - 1)
        for z, y, x in points:
            if lung_mask[z, y, x]:
                control_points.append([z, y, x])

    if not control_points:
        print("Warning: pseudo-regions produced no control points inside the lung mask.")
        return np.empty((0, 3), dtype=float)

    return np.asarray(control_points, dtype=float)


def load_empirical_nodule_distribution(distribution_path):
    if not distribution_path:
        return None
    path = Path(distribution_path)
    if not path.exists():
        print(f"Warning: empirical nodule distribution not found: {path}")
        return None
    dist = np.load(path)
    required = {"relative_zyx", "count_values", "count_probabilities"}
    missing = required.difference(dist.files)
    if missing:
        print(f"Warning: empirical distribution {path} is missing keys: {sorted(missing)}")
        return None
    return {
        "relative_zyx": dist["relative_zyx"].astype(np.float32),
        "count_values": dist["count_values"].astype(np.int64),
        "count_probabilities": dist["count_probabilities"].astype(np.float64),
    }


def build_valid_pseudo_nodule_mask(
    lung_mask,
    erode_iterations=2,
    central_percentile=70,
    min_slice_area_percentile=35,
):
    lung_mask = np.asarray(lung_mask) > 0
    if lung_mask.ndim != 3 or not np.any(lung_mask):
        print("Warning: cannot condition pseudo-nodules on an empty/invalid lung mask.")
        return np.zeros_like(lung_mask, dtype=bool)

    valid_mask = lung_mask.copy()
    erode_iterations = max(0, int(erode_iterations))
    if erode_iterations > 0:
        eroded = binary_erosion(valid_mask, iterations=erode_iterations)
        if np.any(eroded):
            valid_mask = eroded
        else:
            print("Warning: pseudo-nodule lung erosion emptied the mask; using un-eroded lung mask.")

    slice_areas = lung_mask.sum(axis=(1, 2))
    positive_slice_areas = slice_areas[slice_areas > 0]
    if len(positive_slice_areas) > 0:
        min_slice_area_percentile = float(np.clip(min_slice_area_percentile, 0, 100))
        min_slice_area = np.percentile(positive_slice_areas, min_slice_area_percentile)
        valid_slices = slice_areas >= min_slice_area
        slice_filtered = valid_mask & valid_slices[:, None, None]
        if np.any(slice_filtered):
            valid_mask = slice_filtered
        else:
            print("Warning: pseudo-nodule slice-area filtering emptied the mask; keeping previous mask.")

    lung_points = np.argwhere(lung_mask)
    lung_centroid = lung_points.mean(axis=0)
    valid_points = np.argwhere(valid_mask)
    if len(valid_points) > 0:
        central_percentile = float(np.clip(central_percentile, 1, 100))
        lung_distances = np.linalg.norm(lung_points - lung_centroid, axis=1)
        distance_threshold = np.percentile(lung_distances, central_percentile)
        valid_distances = np.linalg.norm(valid_points - lung_centroid, axis=1)
        central_points = valid_points[valid_distances <= distance_threshold]
        min_central_points = 100
        if len(central_points) >= min_central_points:
            central_mask = np.zeros_like(valid_mask, dtype=bool)
            central_mask[central_points[:, 0], central_points[:, 1], central_points[:, 2]] = True
            valid_mask = central_mask
        else:
            print(
                f"Warning: conditioned pseudo-nodule central subset too small "
                f"({len(central_points)} < {min_central_points}); keeping non-central valid mask."
            )

    print(f"Conditioned pseudo-nodule candidate voxels: {int(valid_mask.sum())}")
    return valid_mask


def sample_empirical_pseudo_nodules(
    patient_id,
    volume_shape,
    distribution,
    lung_mask=None,
    min_radius=8,
    max_radius=20,
    valid_erode_iterations=2,
    central_percentile=70,
    min_slice_area_percentile=35,
    position_attempts=100,
    seed_offset=0,
):
    if distribution is None:
        return np.empty((0, 3), dtype=float)

    depth, h, w = volume_shape
    rng = np.random.default_rng(stable_patient_seed(patient_id) + int(seed_offset))
    count_values = distribution["count_values"]
    count_probabilities = distribution["count_probabilities"]
    count_probabilities = count_probabilities / count_probabilities.sum()
    nodule_count = int(rng.choice(count_values, p=count_probabilities))

    relative_zyx = distribution["relative_zyx"]
    if len(relative_zyx) == 0 or nodule_count <= 0:
        return np.empty((0, 3), dtype=float)

    min_radius = max(1, int(min_radius))
    max_radius = max(min_radius, int(max_radius))
    position_attempts = max(1, int(position_attempts))
    offsets = np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
    ], dtype=float)

    control_points = []
    lung_points = None
    valid_mask = None
    valid_points = None
    if lung_mask is not None:
        lung_mask = np.asarray(lung_mask) > 0
        if lung_mask.shape == tuple(volume_shape):
            lung_points = np.argwhere(lung_mask)
            if len(lung_points) == 0:
                lung_points = None
            else:
                valid_mask = build_valid_pseudo_nodule_mask(
                    lung_mask,
                    erode_iterations=valid_erode_iterations,
                    central_percentile=central_percentile,
                    min_slice_area_percentile=min_slice_area_percentile,
                )
                valid_points = np.argwhere(valid_mask)
                if len(valid_points) == 0:
                    print("Warning: conditioned pseudo-nodule mask is empty; falling back to full lung mask.")
                    valid_mask = lung_mask
                    valid_points = lung_points

    accepted_centers = 0
    fallback_centers = 0
    for _ in range(nodule_count):
        center = None
        last_center = None
        for _attempt in range(position_attempts):
            rel_z, rel_y, rel_x = relative_zyx[int(rng.integers(0, len(relative_zyx)))]
            candidate = np.array([
                rel_z * max(depth - 1, 1),
                rel_y * max(h - 1, 1),
                rel_x * max(w - 1, 1),
            ], dtype=float)
            last_center = candidate

            if valid_mask is None:
                center = candidate
                accepted_centers += 1
                break

            z, y, x = np.round(candidate).astype(int)
            z = int(np.clip(z, 0, depth - 1))
            y = int(np.clip(y, 0, h - 1))
            x = int(np.clip(x, 0, w - 1))
            if valid_mask[z, y, x]:
                center = np.array([z, y, x], dtype=float)
                accepted_centers += 1
                break

        if center is None:
            if valid_points is None or len(valid_points) == 0:
                print("Warning: empirical pseudo-nodule sampling found no valid fallback points.")
                continue
            nearest_idx = np.argmin(np.sum((valid_points - last_center) ** 2, axis=1))
            center = valid_points[nearest_idx].astype(float)
            fallback_centers += 1

        radius = float(rng.integers(min_radius, max_radius + 1))
        points = center + offsets * radius
        points[:, 0] = np.clip(points[:, 0], 0, depth - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)
        points[:, 2] = np.clip(points[:, 2], 0, w - 1)
        if lung_mask is not None and lung_mask.shape == tuple(volume_shape):
            snapped_points = []
            for point in points:
                z, y, x = np.round(point).astype(int)
                if valid_mask is not None and valid_mask[z, y, x]:
                    snapped_points.append(point)
                elif lung_mask[z, y, x]:
                    snapped_points.append(point)
                elif valid_points is not None and len(valid_points) > 0:
                    nearest_idx = np.argmin(np.sum((valid_points - point) ** 2, axis=1))
                    snapped_points.append(valid_points[nearest_idx].astype(float))
                elif lung_points is not None and len(lung_points) > 0:
                    nearest_idx = np.argmin(np.sum((lung_points - point) ** 2, axis=1))
                    snapped_points.append(lung_points[nearest_idx].astype(float))
            points = np.asarray(snapped_points, dtype=float)
        control_points.extend(points.tolist())

    print(
        f"Empirical pseudo-nodules for {patient_id}: {nodule_count} nodules, "
        f"{len(control_points)} control points, "
        f"{accepted_centers} conditioned centers, {fallback_centers} nearest-valid fallbacks."
    )
    return np.asarray(control_points, dtype=float)


def fit_surface_grid(
    matrix,
    h,
    w,
    depth,
    num_boundary_anchors,
    rbf_smooth,
    lung_mask=None,
    patient_id=None,
    use_lung_volume_anchors=True,
    lung_anchor_erode_iterations=1,
    snap_surface_to_lung=False,
    anchor_min_lung_area_fraction=0.35,
):
    if len(matrix) > 0:
        target_z = np.mean(matrix[:, 0])
    else:
        target_z = depth // 2

    del patient_id, use_lung_volume_anchors, lung_anchor_erode_iterations
    anchor_z = choose_lung_rich_anchor_z(
        target_z,
        lung_mask,
        min_lung_area_fraction=anchor_min_lung_area_fraction,
    )
    boundary_points = sample_boundary_anchors(anchor_z, h, w, num_boundary_anchors)

    point_labels = (
        ["control"] * len(matrix)
        + ["anchor"] * len(boundary_points)
    )

    if len(boundary_points) > 0:
        matrix = np.vstack([matrix, boundary_points])

    rbf_spline = Rbf(
        matrix[:, 2],
        matrix[:, 1],
        matrix[:, 0],
        function='thin_plate',
        smooth=rbf_smooth,
    )

    x_range = np.arange(w)
    y_range = np.arange(h)
    X, Y = np.meshgrid(x_range, y_range)

    Z_spline_float = rbf_spline(X, Y)
    Z_spline_float = np.clip(Z_spline_float, 0, depth - 1)
    if snap_surface_to_lung and lung_mask is not None:
        Z_spline_float = snap_surface_to_lung_columns(Z_spline_float, lung_mask)
    Z_spline = np.round(Z_spline_float).astype(int)
    return matrix, point_labels, Z_spline_float, Z_spline


def snap_surface_to_lung_columns(Z_spline_float, lung_mask):
    lung_mask = np.asarray(lung_mask) > 0
    if lung_mask.ndim != 3:
        print("Warning: cannot snap surface to lung columns; invalid lung mask shape.")
        return Z_spline_float

    depth, h, w = lung_mask.shape
    if Z_spline_float.shape != (h, w):
        print(
            "Warning: cannot snap surface to lung columns; "
            f"surface shape {Z_spline_float.shape} does not match lung mask {(h, w)}."
        )
        return Z_spline_float

    lung_projection = np.any(lung_mask, axis=0)
    if not np.any(lung_projection):
        print("Warning: cannot snap surface to lung columns; empty lung projection.")
        return Z_spline_float

    z_indices = np.arange(depth)[:, None, None]
    z_min = np.where(lung_mask, z_indices, depth).min(axis=0)
    z_max = np.where(lung_mask, z_indices, -1).max(axis=0)

    snapped = Z_spline_float.copy()
    before_clip = snapped.copy()
    snapped[lung_projection] = np.clip(
        snapped[lung_projection],
        z_min[lung_projection],
        z_max[lung_projection],
    )
    clipped_count = int(np.sum(np.abs(snapped[lung_projection] - before_clip[lung_projection]) > 1e-6))

    rows, cols = np.indices((h, w))
    snapped_int = np.round(snapped).astype(int)
    snapped_int = np.clip(snapped_int, 0, depth - 1)
    inside_lung = lung_mask[snapped_int, rows, cols]
    needs_snap = np.logical_and(lung_projection, ~inside_lung)

    snapped_count = int(np.sum(needs_snap))
    for y, x in np.argwhere(needs_snap):
        valid_z = np.flatnonzero(lung_mask[:, y, x])
        if len(valid_z) == 0:
            continue
        current_z = snapped_int[y, x]
        nearest_z = valid_z[np.argmin(np.abs(valid_z - current_z))]
        snapped[y, x] = float(nearest_z)

    coverage_after = float(
        np.mean(lung_mask[np.round(snapped).astype(int), rows, cols][lung_projection])
    )
    print(
        "Surface lung-column snap: "
        f"{clipped_count} clipped, {snapped_count} nearest-valid adjusted; "
        f"lung-projection coverage={coverage_after:.3f}"
    )
    return snapped


def compute_lung_coverage(Z_spline, lung_mask):
    lung_mask = np.asarray(lung_mask) > 0
    h, w = Z_spline.shape
    rows, cols = np.indices((h, w))
    lung_projection = np.any(lung_mask, axis=0)
    if not np.any(lung_projection):
        print("Warning: empty lung projection; falling back to full-image lung coverage.")
        return float(np.mean(lung_mask[Z_spline, rows, cols]))
    sampled_inside_lung = lung_mask[Z_spline, rows, cols]
    return float(np.mean(sampled_inside_lung[lung_projection]))


def sample_contour_points(mask_2d, num_points):
    if num_points <= 0:
        return np.empty((0, 2), dtype=float)

    contour = np.logical_and(mask_2d, ~binary_erosion(mask_2d))
    y_coords, x_coords = np.where(contour)
    if len(y_coords) == 0:
        y_coords, x_coords = np.where(mask_2d)
    if len(y_coords) == 0:
        return np.empty((0, 2), dtype=float)

    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)
    angles = np.arctan2(y_coords - center_y, x_coords - center_x)
    order = np.argsort(angles)
    y_coords = y_coords[order]
    x_coords = x_coords[order]

    sample_indices = np.linspace(0, len(y_coords) - 1, num=min(num_points, len(y_coords)), dtype=int)
    return np.column_stack([y_coords[sample_indices], x_coords[sample_indices]]).astype(float)


def sample_lung_volume_anchors(
    lung_mask,
    num_anchors,
    patient_id=None,
    erode_iterations=1,
    target_z=None,
    min_slice_area_fraction=0.35,
):
    if num_anchors <= 0:
        return np.empty((0, 3), dtype=float)

    lung_mask = np.asarray(lung_mask) > 0
    if lung_mask.ndim != 3 or not np.any(lung_mask):
        print("Warning: cannot sample lung volume anchors from an empty/invalid lung mask.")
        return np.empty((0, 3), dtype=float)

    anchor_mask = lung_mask
    if erode_iterations > 0:
        eroded = binary_erosion(lung_mask, iterations=int(erode_iterations))
        if np.any(eroded):
            anchor_mask = eroded
        else:
            print("Warning: eroded lung mask is empty for anchors; using un-eroded lung mask.")

    slice_areas = anchor_mask.sum(axis=(1, 2))
    valid_z = np.flatnonzero(slice_areas > 0)
    if len(valid_z) == 0:
        print("Warning: no lung voxels available for lung volume anchors.")
        return np.empty((0, 3), dtype=float)

    rng = np.random.default_rng(stable_patient_seed(patient_id) + 7919)
    _, h, w = anchor_mask.shape
    anchor_points = np.argwhere(anchor_mask)
    mid_x = float(np.median(anchor_points[:, 2]))
    midline_gap = max(2, int(round(0.08 * w)))

    max_slice_area = float(np.max(slice_areas[valid_z]))
    min_anchor_slice_area = max_slice_area * float(np.clip(min_slice_area_fraction, 0.0, 1.0))
    robust_z = valid_z[slice_areas[valid_z] >= min_anchor_slice_area]

    if target_z is None:
        z = int(valid_z[np.argmax(slice_areas[valid_z])])
    else:
        target_z_float = float(target_z)
        nearest_z = int(valid_z[np.argmin(np.abs(valid_z - target_z_float))])
        if slice_areas[nearest_z] >= min_anchor_slice_area or len(robust_z) == 0:
            z = nearest_z
        else:
            z = int(robust_z[np.argmin(np.abs(robust_z - target_z_float))])
            print(
                "Warning: anchor target slice has small lung area "
                f"({int(slice_areas[nearest_z])} voxels at z={nearest_z}); "
                f"using robust z={z} ({int(slice_areas[z])} voxels)."
            )

    def sample_anchors_from_slice_mask(slice_mask, count):
        labeled_slice, num_features = nd_label(slice_mask)
        if num_features == 0 or count <= 0:
            return []

        component_sizes = np.bincount(labeled_slice.ravel())
        component_sizes[0] = 0
        keep_labels = np.argsort(component_sizes)[-2:]
        keep_labels = keep_labels[component_sizes[keep_labels] >= 20]
        if len(keep_labels) == 0:
            keep_labels = np.array([int(np.argmax(component_sizes))])

        component_label = int(keep_labels[np.argmax(component_sizes[keep_labels])])
        component_mask = labeled_slice == component_label
        eroded_component = binary_erosion(component_mask, iterations=1)
        if np.any(eroded_component):
            component_mask = eroded_component

        yx_points = np.argwhere(component_mask)
        if len(yx_points) == 0:
            return []

        centroid = yx_points.mean(axis=0)
        distances = np.linalg.norm(yx_points - centroid, axis=1)
        central_count = max(1, int(np.ceil(0.50 * len(yx_points))))
        central_indices = np.argpartition(distances, central_count - 1)[:central_count]
        replace = len(central_indices) < count
        selected_indices = rng.choice(central_indices, size=count, replace=replace)
        return [
            [float(z), float(y), float(x)]
            for y, x in yx_points[selected_indices]
        ]

    slice_mask = anchor_mask[z]
    x_coords = np.arange(w)[None, :]
    left_mask = np.logical_and(slice_mask, x_coords < (mid_x - midline_gap))
    right_mask = np.logical_and(slice_mask, x_coords > (mid_x + midline_gap))

    anchors = []
    if np.any(left_mask) and np.any(right_mask):
        left_count = int(np.ceil(num_anchors / 2))
        right_count = int(num_anchors - left_count)
        anchors.extend(sample_anchors_from_slice_mask(left_mask, left_count))
        anchors.extend(sample_anchors_from_slice_mask(right_mask, right_count))
    elif np.any(left_mask):
        print(f"Warning: no right lateral lung voxels at z={z}; using left lung anchors only.")
        anchors.extend(sample_anchors_from_slice_mask(left_mask, num_anchors))
    elif np.any(right_mask):
        print(f"Warning: no left lateral lung voxels at z={z}; using right lung anchors only.")
        anchors.extend(sample_anchors_from_slice_mask(right_mask, num_anchors))
    else:
        print(f"Warning: no lateral lung voxels at z={z}; using full slice mask for anchors.")
        anchors.extend(sample_anchors_from_slice_mask(slice_mask, num_anchors))

    anchors = np.unique(np.asarray(anchors, dtype=float), axis=0)
    if len(anchors) > 0:
        print(
            "Lung volume anchors: "
            f"{len(anchors)} points on z={z} "
            f"(target_z={float(target_z) if target_z is not None else z:.2f})"
        )
    return anchors


def choose_lung_rich_anchor_z(target_z, lung_mask, min_lung_area_fraction=0.35):
    if lung_mask is None:
        return float(target_z)

    lung_mask = np.asarray(lung_mask) > 0
    if lung_mask.ndim != 3 or not np.any(lung_mask):
        return float(target_z)

    slice_areas = lung_mask.sum(axis=(1, 2))
    valid_z = np.flatnonzero(slice_areas > 0)
    if len(valid_z) == 0:
        return float(target_z)

    target_z_float = float(target_z)
    nearest_z = int(valid_z[np.argmin(np.abs(valid_z - target_z_float))])
    max_slice_area = float(np.max(slice_areas[valid_z]))
    min_slice_area = max_slice_area * float(np.clip(min_lung_area_fraction, 0.0, 1.0))
    if slice_areas[nearest_z] >= min_slice_area:
        return target_z_float

    robust_z = valid_z[slice_areas[valid_z] >= min_slice_area]
    if len(robust_z) == 0:
        return target_z_float

    anchor_z = float(robust_z[np.argmin(np.abs(robust_z - target_z_float))])
    print(
        "Anchor box target_z moved to lung-rich slice: "
        f"{target_z_float:.2f} -> {anchor_z:.0f} "
        f"(slice lung area {int(slice_areas[nearest_z])} < {int(min_slice_area)})."
    )
    return anchor_z


def sample_boundary_anchors(target_z, h, w, num_anchors=None):
    del num_anchors
    if h <= 0 or w <= 0:
        return np.empty((0, 3), dtype=float)

    anchors = []
    edge_coords = [0, w // 2, w - 1]
    for x_edge in edge_coords:
        for y_edge in [0, h - 1]:
            anchors.append([target_z, y_edge, x_edge])
    for y_edge in [h // 2]:
        for x_edge in [0, w - 1]:
            anchors.append([target_z, y_edge, x_edge])

    print(f"Using flat boundary anchor box at z={target_z:.2f}.")
    return np.asarray(anchors, dtype=float)


def _load_volume(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Reading LUNA16 .nii/.nii.gz files requires nibabel.") from exc

    volume = nib.load(str(path)).get_fdata().astype(np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume at {path}, got shape {volume.shape}.")
    return volume.transpose(2, 1, 0)


class LUNA16NiftiDataset(Dataset):
    """Dataset for LUNA16 CSV files with seriesuid/image_path/nodule_mask_path columns."""

    def __init__(
        self,
        csv_file,
        processed_dir,
        split=None,
        return_mask=True,
        return_lung_mask=True,
        only_no_nodules=False,
    ):
        self.data_frame = pd.read_csv(csv_file)
        if split and "split" in self.data_frame.columns:
            self.data_frame = self.data_frame[self.data_frame["split"].astype(str) == split].reset_index(drop=True)
        if only_no_nodules:
            if "nodule_count" not in self.data_frame.columns:
                raise ValueError("data.only_no_nodules=true requires a CSV with a nodule_count column.")
            self.data_frame = self.data_frame[self.data_frame["nodule_count"].fillna(0).astype(int) == 0].reset_index(drop=True)
        self.processed_dir = Path(processed_dir)
        self.return_mask = return_mask
        self.return_lung_mask = return_lung_mask

    def __len__(self):
        return len(self.data_frame)

    def _resolve(self, row, column, suffix):
        raw = str(row[column]) if column in row and pd.notna(row[column]) else ""
        candidates = []
        if raw:
            path = Path(raw)
            candidates.append(path)
            if not path.is_absolute():
                candidates.append(self.processed_dir / path)
                if len(path.parts) > 1 and path.parts[0].startswith("subset"):
                    candidates.append(self.processed_dir / Path(*path.parts[1:]))
        seriesuid = str(row["seriesuid"]) if "seriesuid" in row and pd.notna(row["seriesuid"]) else ""
        if seriesuid:
            candidates.append(self.processed_dir / seriesuid / f"{seriesuid}_{suffix}.nii.gz")
            candidates.append(self.processed_dir / seriesuid / f"{seriesuid}_{suffix}.npy")
            for subset_idx in range(10):
                candidates.append(self.processed_dir / f"subset{subset_idx}" / seriesuid / f"{seriesuid}_{suffix}.nii.gz")
                candidates.append(self.processed_dir / f"subset{subset_idx}" / seriesuid / f"{seriesuid}_{suffix}.npy")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else Path(raw)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        patient_id = str(row["seriesuid"] if "seriesuid" in row else row["patient_id"])
        target = int(row["target"]) if "target" in row and pd.notna(row["target"]) else 0
        volume = torch.from_numpy(_load_volume(self._resolve(row, "image_path", "volume"))).float()
        if volume.ndim == 3:
            volume = volume.unsqueeze(1)
        label = torch.tensor([target], dtype=torch.float32)

        outputs = [volume, label, patient_id]

        if self.return_mask:
            mask_path = self._resolve(row, "nodule_mask_path", "nodule_mask")
            if mask_path.exists():
                mask = torch.from_numpy((_load_volume(mask_path) > 0).astype(np.uint8)).long()
            else:
                mask = torch.zeros(volume.shape[0], volume.shape[2], volume.shape[3]).long()
            outputs.append(mask)

        if self.return_lung_mask:
            lung_mask_path = self._resolve(row, "lung_mask_path", "lung_mask")
            if lung_mask_path.exists():
                lung_mask = torch.from_numpy((_load_volume(lung_mask_path) > 0).astype(np.uint8)).long()
                if tuple(lung_mask.shape) != (volume.shape[0], volume.shape[2], volume.shape[3]):
                    print(
                        f"Warning: lung mask shape mismatch for {patient_id}: "
                        f"{tuple(lung_mask.shape)} vs volume {(volume.shape[0], volume.shape[2], volume.shape[3])}."
                    )
                    lung_mask = torch.zeros(volume.shape[0], volume.shape[2], volume.shape[3]).long()
            else:
                print(f"Warning: missing saved lung mask for {patient_id}: {lung_mask_path}")
                lung_mask = torch.zeros(volume.shape[0], volume.shape[2], volume.shape[3]).long()
            outputs.append(lung_mask)

        return tuple(outputs)

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
    saliency_cfg = cfg.get("saliency", {})
    num_contour_points = saliency_cfg.get("num_contour_points", 4)
    num_boundary_anchors = saliency_cfg.get("num_boundary_anchors", 8)
    lungmask_model_name = saliency_cfg.get("lungmask_model_name", "R231")
    lungmask_force_cpu = saliency_cfg.get("lungmask_force_cpu", False)
    lung_mask_method = saliency_cfg.get("lung_mask_method", "body_threshold")
    normalized_air_threshold = saliency_cfg.get("normalized_air_threshold", 0.35)
    hu_air_min = saliency_cfg.get("hu_air_min", -1000)
    hu_air_max = saliency_cfg.get("hu_air_max", -320)
    body_threshold_percentile = saliency_cfg.get("body_threshold_percentile", 1)
    lung_component_count = saliency_cfg.get("lung_component_count", 2)
    pseudo_min_regions = saliency_cfg.get("pseudo_min_regions", 1)
    pseudo_max_regions = saliency_cfg.get("pseudo_max_regions", 3)
    pseudo_min_radius = saliency_cfg.get("pseudo_min_radius", 8)
    pseudo_max_radius = saliency_cfg.get("pseudo_max_radius", 20)
    pseudo_erode_iterations = saliency_cfg.get("pseudo_erode_iterations", 2)
    pseudo_empirical_position_attempts = saliency_cfg.get("pseudo_empirical_position_attempts", 100)
    pseudo_min_slice_area_percentile = saliency_cfg.get("pseudo_min_slice_area_percentile", 35)
    lung_window_center = saliency_cfg.get("lung_window_center", -600)
    lung_window_width = saliency_cfg.get("lung_window_width", 1500)
    pseudo_central_percentile = saliency_cfg.get("pseudo_central_percentile", 70)
    min_lung_coverage = saliency_cfg.get("min_lung_coverage", 0.25)
    pseudo_max_attempts = saliency_cfg.get("pseudo_max_attempts", 5)
    rbf_smooth = saliency_cfg.get("rbf_smooth", 0.1)
    min_best_lung_coverage = saliency_cfg.get("min_best_lung_coverage", 0.10)
    use_lung_volume_anchors = saliency_cfg.get("use_lung_volume_anchors", True)
    lung_anchor_erode_iterations = saliency_cfg.get("lung_anchor_erode_iterations", 1)
    snap_surface_to_lung = saliency_cfg.get("snap_surface_to_lung", False)
    anchor_min_lung_area_fraction = saliency_cfg.get("anchor_min_lung_area_fraction", 0.35)
    use_saved_lung_masks = saliency_cfg.get("use_saved_lung_masks", True)
    empirical_distribution_path = saliency_cfg.get(
        "empirical_nodule_distribution_path",
        "outputs/luna16_saliency_control_point_distribution/empirical_nodule_distribution_from_control_points.npz",
    )
    use_empirical_pseudo_nodules = saliency_cfg.get("use_empirical_pseudo_nodules", True)
    empirical_distribution = load_empirical_nodule_distribution(empirical_distribution_path) if use_empirical_pseudo_nodules else None

    os.makedirs(save_path, exist_ok=True)

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
            if len(sample) == 5:
                img, label, patient_id, mask, saved_lung_mask = sample
            elif len(sample) == 4:
                img, label, patient_id, mask = sample
                saved_lung_mask = None
            else:
                img, label, patient_id = sample
                mask = None
                saved_lung_mask = None

            print(f"Processing patient: {patient_id}")

            # get width and height from img
            h, w = img.shape[2:]
            print(h, w)
            volume_np = img.numpy()[:, 0, :, :]
            lung_mask = None
            if use_saved_lung_masks and saved_lung_mask is not None:
                saved_lung_mask_np = (saved_lung_mask.numpy() > 0)
                if np.any(saved_lung_mask_np):
                    lung_mask = saved_lung_mask_np
                    print(f"Using saved lung mask for {patient_id}: {int(lung_mask.sum())} voxels.")
                else:
                    print(f"Warning: saved lung mask is empty for {patient_id}; falling back to runtime lung mask.")

            # Save every sample directly under one output root, keyed by ID.
            current_save_dir = os.path.join(save_path, patient_id)
            os.makedirs(current_save_dir, exist_ok=True)

            save_file_path = os.path.join(current_save_dir, f"surface_{patient_id}.png")
            save_surface_grid_path = os.path.join(current_save_dir, f"surface_grid_{patient_id}.npy")

            save_surface_grid = cfg.get("saliency", {}).get("save_surface_grid", False)
            if os.path.exists(save_file_path) and (not save_surface_grid or os.path.exists(save_surface_grid_path)):
                continue

            matrix = None
            Z_spline = None
            if mask is not None: # and label.item() == 1:
                # Malignant case with mask: Use centroids of each separate nodule
                mask_np = (mask.numpy() > 0).astype(int)
                
                # Label connected components (nodules)
                labeled_mask, num_features = nd_label(mask_np)
                
                if num_features > 0:
                    # Remove overlapping nodules (keep only the larger ones)
                    print(f"Found {num_features} nodules for patient {patient_id}")
                    labeled_mask, kept_labels = remove_overlapping_nodules(labeled_mask, num_features)
                    
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
                        
                        # 3. Campiona punti di controllo da QUELLA slice specifica:
                        # centro + punti sul contorno per "appiattire" lo spline sul nodulo.
                        y_coords_at_z = coords[1][z_coords == best_z]
                        x_coords_at_z = coords[2][z_coords == best_z]
                        mask_at_z = nodule_mask[best_z]
                        contour_points = sample_contour_points(mask_at_z, num_contour_points)

                        points_to_add = [[best_z, np.mean(y_coords_at_z), np.mean(x_coords_at_z)]]
                        points_to_add.extend([[best_z, y, x] for y, x in contour_points])
                        matrix_list.extend(points_to_add)

                    matrix = np.array(matrix_list)
                    print(f"Using {len(matrix)} nodule centers from mask.")
                else:
                    print(f"Warning: No nodules found in mask for {patient_id}.")
            
            if matrix is None:
                print(f"No nodule mask for {patient_id}: sampling pseudo-nodules.")
                if lung_mask is None:
                    lung_mask = get_lung_mask(
                        volume_np,
                        model_name=lungmask_model_name,
                        force_cpu=lungmask_force_cpu,
                        method=lung_mask_method,
                        normalized_air_threshold=normalized_air_threshold,
                        hu_air_min=hu_air_min,
                        hu_air_max=hu_air_max,
                        body_threshold_percentile=body_threshold_percentile,
                        lung_component_count=lung_component_count,
                    )

                pseudo_max_attempts = max(1, int(pseudo_max_attempts))
                best_candidate = None
                best_lung_coverage = -1.0
                for attempt in range(pseudo_max_attempts):
                    print(f"Attempt: {attempt + 1}/{pseudo_max_attempts}")
                    if empirical_distribution is not None:
                        candidate_matrix = sample_empirical_pseudo_nodules(
                            patient_id,
                            volume_np.shape,
                            empirical_distribution,
                            lung_mask=lung_mask,
                            min_radius=pseudo_min_radius,
                            max_radius=pseudo_max_radius,
                            valid_erode_iterations=pseudo_erode_iterations,
                            central_percentile=pseudo_central_percentile,
                            min_slice_area_percentile=pseudo_min_slice_area_percentile,
                            position_attempts=pseudo_empirical_position_attempts,
                            seed_offset=attempt,
                        )
                    else:
                        candidate_matrix = sample_pseudo_regions_inside_lung(
                            patient_id,
                            lung_mask,
                            min_regions=pseudo_min_regions,
                            max_regions=pseudo_max_regions,
                            min_radius=pseudo_min_radius,
                            max_radius=pseudo_max_radius,
                            erode_iterations=pseudo_erode_iterations,
                            central_percentile=pseudo_central_percentile,
                            seed_offset=attempt,
                        )
                    if len(candidate_matrix) == 0:
                        print("Rejected: no pseudo-region control points.")
                        continue

                    candidate_matrix, point_labels, candidate_Z_spline_float, candidate_Z_spline = fit_surface_grid(
                        candidate_matrix,
                        h,
                        w,
                        img.shape[0],
                        num_boundary_anchors,
                        rbf_smooth,
                        lung_mask=lung_mask,
                        patient_id=patient_id,
                        use_lung_volume_anchors=use_lung_volume_anchors,
                        lung_anchor_erode_iterations=lung_anchor_erode_iterations,
                        snap_surface_to_lung=snap_surface_to_lung,
                        anchor_min_lung_area_fraction=anchor_min_lung_area_fraction,
                    )
                    lung_coverage = compute_lung_coverage(candidate_Z_spline, lung_mask)
                    print(f"Lung coverage: {lung_coverage:.3f}")
                    if lung_coverage > best_lung_coverage:
                        best_lung_coverage = lung_coverage
                        best_candidate = (
                            candidate_matrix,
                            point_labels,
                            candidate_Z_spline_float,
                            candidate_Z_spline,
                        )
                    if lung_coverage >= min_lung_coverage:
                        print("Accepted")
                        matrix = candidate_matrix
                        point_labels = point_labels
                        Z_spline_float = candidate_Z_spline_float
                        Z_spline = candidate_Z_spline
                        break
                    print("Rejected")

                if matrix is None:
                    if best_candidate is not None and best_lung_coverage >= min_best_lung_coverage:
                        print(
                            f"Warning: no attempt reached min_lung_coverage={min_lung_coverage:.3f}; "
                            f"using best attempt with lung coverage {best_lung_coverage:.3f}."
                        )
                        matrix, point_labels, Z_spline_float, Z_spline = best_candidate
                    else:
                        print(f"Warning: falling back to mid-slice plane for {patient_id}.")
                        matrix = mid_slice_plane(img.shape[0], h, w)

            if Z_spline is None:
                if use_lung_volume_anchors and lung_mask is None:
                    lung_mask = get_lung_mask(
                        volume_np,
                        model_name=lungmask_model_name,
                        force_cpu=lungmask_force_cpu,
                        method=lung_mask_method,
                        normalized_air_threshold=normalized_air_threshold,
                        hu_air_min=hu_air_min,
                        hu_air_max=hu_air_max,
                        body_threshold_percentile=body_threshold_percentile,
                        lung_component_count=lung_component_count,
                    )
                matrix, point_labels, Z_spline_float, Z_spline = fit_surface_grid(
                    matrix,
                    h,
                    w,
                    img.shape[0],
                    num_boundary_anchors,
                    rbf_smooth,
                    lung_mask=lung_mask,
                    patient_id=patient_id,
                    use_lung_volume_anchors=use_lung_volume_anchors,
                    lung_anchor_erode_iterations=lung_anchor_erode_iterations,
                    snap_surface_to_lung=snap_surface_to_lung,
                    anchor_min_lung_area_fraction=anchor_min_lung_area_fraction,
                )
            if save_surface_grid:
                if save_surface_grid:
                    np.save(
                        os.path.join(current_save_dir, f"surface_grid_float_{patient_id}.npy"),
                        Z_spline_float.astype(np.float32)
                    )

                    np.save(
                        os.path.join(current_save_dir, f"surface_grid_int_{patient_id}.npy"),
                        Z_spline.astype(np.int16)
                    )

                    np.save(
                        os.path.join(current_save_dir, f"control_points_{patient_id}.npy"),
                        matrix.astype(np.float32)
                    )

                    np.save(
                        os.path.join(current_save_dir, f"point_labels_{patient_id}.npy"),
                        np.asarray(point_labels)
                    )
            
            saliency_cfg = cfg.get("saliency", {})
            debug_plot = saliency_cfg.get("debug", False) or saliency_cfg.get("show_plot", False)
            if debug_plot:
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

            img_np = img.numpy() # Shape: (D, 1, H, W)
            rows, cols = np.indices((h, w))
            output_image = img_np[Z_spline, 0, rows, cols]

            # LUNA16 preprocessing stores lung-windowed intensities as uint8 [0, 255].
            # Convert them back to float [0, 1] for PNG export.
            if output_image.min() >= 0 and output_image.max() <= 255:
                output_image = output_image.astype(np.float32) / 255.0

            else:
                # Fallback only for datasets genuinely stored in HU.
                window_min = lung_window_center - (lung_window_width / 2)
                window_max = lung_window_center + (lung_window_width / 2)

                output_image = np.clip(output_image, window_min, window_max)
                output_image = (output_image - window_min) / (window_max - window_min)

            output_image = np.clip(output_image, 0, 1)
            Image.fromarray((output_image * 255).astype(np.uint8)).convert("RGB").save(save_file_path)
        except Exception as e:
            print(f"\nERROR ITEM {k}")
            print(f"PATIENT: {patient_id}")
            traceback.print_exc()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Ensure return_mask is True for this task
    OmegaConf.set_struct(cfg, False)
    cfg.data.return_mask = True
    
    print("Building datasets...")
    if cfg.data.get("dataset_type") == "luna16_nii":
        only_no_nodules = cfg.data.get("only_no_nodules", False)
        return_lung_mask = cfg.data.get("return_lung_mask", True)
        train_dataset = LUNA16NiftiDataset(
            cfg.data.csv_file,
            cfg.data.processed_dir,
            split='train',
            return_mask=True,
            return_lung_mask=return_lung_mask,
            only_no_nodules=only_no_nodules,
        )
        val_dataset = LUNA16NiftiDataset(
            cfg.data.csv_file,
            cfg.data.processed_dir,
            split='val',
            return_mask=True,
            return_lung_mask=return_lung_mask,
            only_no_nodules=only_no_nodules,
        )
        test_dataset = LUNA16NiftiDataset(
            cfg.data.csv_file,
            cfg.data.processed_dir,
            split='test',
            return_mask=True,
            return_lung_mask=return_lung_mask,
            only_no_nodules=only_no_nodules,
        )
    else:
        try:
            import src.builder as builder
            import src.datamodule as datamodule
        except ModuleNotFoundError:
            import src.cls.builder as builder
            import src.cls.datamodule as datamodule

        # Instantiate DataModule to validate/instantiate configured transforms.
        dm = datamodule.DataModule(cfg.data)
        train_dataset = builder.build_dataset(cfg.data, split='train', transforms=None)
        val_dataset = builder.build_dataset(cfg.data, split='val', transforms=None)
        test_dataset = builder.build_dataset(cfg.data, split='test', transforms=None)

    save_path = cfg.get("saliency", {}).get("save_path", "Lung2Dsynt_gt_nodules/")
    print(f"Path where surfaces will be saved: {save_path}")

    if train_dataset:
        print("Processing training set...")
        save_surfaces(save_path, train_dataset, cfg)
    if val_dataset:
        print("Processing validation set...")
        save_surfaces(save_path, val_dataset, cfg)
    if test_dataset:
        print("Processing test set...")
        save_surfaces(save_path, test_dataset, cfg)

if __name__ == "__main__":
    main()
