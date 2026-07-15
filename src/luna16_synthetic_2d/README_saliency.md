# LUNA16 Saliency Surface Generation

This README explains how `saliency.py` generates synthetic 2D CT surfaces from LUNA16 3D volumes, both when a scan has annotated nodules and when it has no nodules.

## Entry Points

Main Python entry point:

```bash
python3 src/luna16_synthetic_2d/saliency.py --config-path src/luna16_synthetic_2d/conf --config-name config
```

Convenience wrapper:

```bash
bash bash/run_luna16_saliency.sh
```

The bash script passes Hydra overrides for the processed LUNA16 directory, CV split CSV, output path, lung-mask behavior, RBF/TPS parameters, pseudo-nodule sampling, and QC thresholds.

## High-Level Pipeline

For each selected fold, `saliency.py` builds train/val/test datasets from the fold CSV and processes every scan.

For each scan:

1. Load the 3D CT volume as `[D, H, W]`.
2. Load the nodule mask if available.
3. Load a saved lung mask if configured and present.
4. Build control points:
   - from real nodule masks for nodule-positive scans;
   - from sampled pseudo-nodules inside the lung for no-nodule scans.
5. Add boundary/anchor points to stabilize the surface.
6. Fit a thin-plate RBF surface `z = f(x, y)`.
7. Sample the 3D CT volume along that surface to produce a synthetic 2D image.
8. Save the PNG and, optionally, the surface grid/control points.

## Dataset Loading

`LUNA16NiftiDataset` reads a classification CSV with at least `seriesuid`, `split`, and path columns such as:

- `image_path`
- `nodule_mask_path`
- `lung_mask_path`
- `nodule_count`

The dataset resolves files either from explicit CSV paths or by searching under `data.processed_dir`.

When `data.only_no_nodules=true`, the dataset keeps only rows with:

```text
nodule_count == 0
```

That mode is useful for generating synthetic negative/no-nodule surfaces.

## Case 1: Scans With Nodules

If a nodule mask exists and contains connected components, `saliency.py` uses the real annotated nodules as control points.

The flow is:

1. Convert the 3D nodule mask to binary.
2. Label connected components with `scipy.ndimage.label`.
3. Remove overlapping nodules in XY projection, keeping the larger component.
4. For each remaining nodule:
   - find the axial slice `z` where the nodule has maximum area;
   - compute the nodule center on that slice;
   - sample contour points from the 2D nodule mask on that same slice.
5. Build the control-point matrix:

```text
[z, y, x]
```

Each nodule contributes:

```text
1 center point + saliency.num_contour_points contour points
```

Default:

```yaml
saliency.num_contour_points: 4
```

So each nodule usually contributes 5 control points.

These points guide the fitted surface so that the final 2D image passes through, or near, the annotated nodule locations.

## Case 2: Scans Without Nodules

If no valid nodule mask is found, `saliency.py` samples pseudo-nodule control points inside the lung.

This happens for true no-nodule scans, and also for scans where the mask is missing or empty.

The code first tries to use a saved lung mask. If no usable saved mask exists, it estimates one from the CT volume using `get_lung_mask`.

The runtime lung mask is deterministic and body-constrained:

1. Estimate a body mask.
2. Identify air-like voxels:
   - HU mode: `hu_air_min <= voxel <= hu_air_max`;
   - normalized mode: `voxel <= normalized_air_threshold`.
3. Intersect air voxels with the body mask.
4. Keep the largest lung components.

### Empirical Pseudo-Nodules

By default, no-nodule sampling uses an empirical distribution:

```yaml
saliency.use_empirical_pseudo_nodules: true
saliency.empirical_nodule_distribution_path: outputs/luna16_saliency_control_point_distribution/empirical_nodule_distribution_from_control_points.npz
```

That file provides:

- empirical relative nodule positions;
- empirical nodule-count probabilities.

For each no-nodule scan:

1. Sample how many pseudo-nodules to create.
2. Sample relative positions from the empirical distribution.
3. Convert relative positions to absolute `[z, y, x]`.
4. Accept only positions inside a conditioned lung mask.
5. If a sampled point is invalid, fall back to the nearest valid lung point.
6. Around each pseudo-nodule center, create center/neighbor control points using a sampled radius.

The conditioned lung mask favors central, sufficiently large lung slices and can be eroded to avoid boundary artifacts.

### Non-Empirical Fallback

If the empirical distribution is disabled or missing, the code uses `sample_pseudo_regions_inside_lung`.

That path:

1. Computes the lung centroid.
2. Optionally erodes the lung mask.
3. Keeps central lung candidate voxels.
4. Randomly samples `pseudo_min_regions..pseudo_max_regions`.
5. Creates local control points around each sampled center.

Sampling is deterministic per patient because it uses a stable seed from `seriesuid`.

## Surface Fitting

Both nodule and no-nodule paths eventually call `fit_surface_grid`.

The fitted surface is a thin-plate RBF:

```python
Rbf(x, y, z, function="thin_plate", smooth=rbf_smooth)
```

Inputs:

- nodule or pseudo-nodule control points;
- boundary anchors from `sample_boundary_anchors`;
- optional lung-aware anchor slice selection.

The result is:

- `surface_grid_float_<seriesuid>.npy`: float `z` surface;
- `surface_grid_int_<seriesuid>.npy`: rounded integer `z` surface;
- `control_points_<seriesuid>.npy`: final control/anchor points;
- `point_labels_<seriesuid>.npy`: labels such as `control` and `anchor`.

The 2D image is produced by sampling:

```text
output_image[y, x] = volume[z_surface[y, x], y, x]
```

## No-Nodule QC And Fallbacks

For pseudo-nodule/no-nodule scans, the code tries multiple pseudo-control-point samples:

```yaml
saliency.pseudo_max_attempts: 5
saliency.min_lung_coverage: 0.25
saliency.min_best_lung_coverage: 0.10
```

For each attempt:

1. Generate pseudo-control points.
2. Fit a surface.
3. Compute how much of the sampled surface lies inside the lung projection.
4. Accept the first attempt with lung coverage above `min_lung_coverage`.

If no attempt reaches `min_lung_coverage`, the best attempt is kept only if it reaches `min_best_lung_coverage`.

If all attempts fail, the code falls back to a flat mid-slice plane.

## Saved Outputs

For each patient, files are saved under:

```text
<saliency.save_path>/<seriesuid>/
```

Typical outputs:

```text
surface_<seriesuid>.png
surface_grid_float_<seriesuid>.npy
surface_grid_int_<seriesuid>.npy
control_points_<seriesuid>.npy
point_labels_<seriesuid>.npy
```

The PNG is a lung-windowed synthetic 2D image. If the source volume is already uint8-like `[0, 255]`, it is normalized to `[0, 1]`; otherwise the configured lung window is applied.

## Important Configuration

Common Hydra config values:

```yaml
data:
  dataset_type: luna16_nii
  processed_dir: data/LUNA16_preprocessed
  csv_file: data/LUNA16_preprocessed/cv_splits/luna16_classification_fold0.csv
  return_mask: true
  return_lung_mask: true
  only_no_nodules: false

saliency:
  save_path: outputs/luna16_saliency_synthetic_gt
  save_surface_grid: true
  rbf_smooth: 0.1
  num_boundary_anchors: 24
  num_contour_points: 4
  use_saved_lung_masks: true
  use_empirical_pseudo_nodules: true
  pseudo_max_attempts: 5
  min_lung_coverage: 0.25
  min_best_lung_coverage: 0.10
```

To generate only no-nodule surfaces through the bash wrapper:

```bash
ONLY_NO_NODULES=true bash bash/run_luna16_saliency.sh
```

To process both nodule and no-nodule rows:

```bash
ONLY_NO_NODULES=false bash bash/run_luna16_saliency.sh
```

## Practical Interpretation

For nodule-positive scans, the synthetic 2D surface is supervised by real nodule masks.

For no-nodule scans, the synthetic 2D surface is not random flat slicing: it is guided by pseudo-control points sampled inside plausible lung regions, optionally following the empirical spatial/count distribution learned from real nodule control points. This makes negative/no-nodule synthetic images structurally comparable to positive synthetic images while avoiding real annotated nodules.
