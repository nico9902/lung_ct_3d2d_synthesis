# Empirical LUNA16 Nodule Position Distribution From Control Points

Source output root: `outputs/luna16_saliency_synthetic_gt`

This distribution is derived only from `control_points_*.npy`, `point_labels_*.npy`, and surface-grid files saved in the output folders.
Every 5 control points are treated as one nodule; the first point in each group is the nodule center used for the empirical position distribution.
It does **not** use `LUNA16_preprocessed` metadata or annotation CSV coordinates.

## Summary

- Samples with control points: **601**
- Control points: **5895**
- Nodules: **1179**
- Positive-log filter enabled: **True**
- Positive scans in log: **601**
- Skipped because not positive in log: **287**
- Skipped samples: **0**

## Relative Position Distribution

Nodule-center coordinates are normalized per saved surface grid and saved in `[rel_z, rel_y, rel_x]`.

| Axis | Mean | Std | Q05 | Q25 | Q50 | Q75 | Q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rel_z | 0.827 | 0.209 | 0.378 | 0.706 | 0.946 | 0.969 | 1.000 |
| rel_y | 0.564 | 0.219 | 0.185 | 0.392 | 0.598 | 0.748 | 0.872 |
| rel_x | 0.483 | 0.280 | 0.116 | 0.226 | 0.381 | 0.761 | 0.893 |

## Nodule Count Distribution Per Scan

| Nodules per scan | Scan count | Probability |
|---:|---:|---:|
| 1 | 313 | 0.5208 |
| 2 | 156 | 0.2596 |
| 3 | 64 | 0.1065 |
| 4 | 24 | 0.0399 |
| 5 | 24 | 0.0399 |
| 6 | 9 | 0.0150 |
| 7 | 4 | 0.0067 |
| 8 | 2 | 0.0033 |
| 9 | 4 | 0.0067 |
| 12 | 1 | 0.0017 |

## Files

- Nodule positions CSV: `outputs/luna16_saliency_control_point_distribution/empirical_nodule_positions_from_control_points.csv`
- Control point positions CSV: `outputs/luna16_saliency_control_point_distribution/empirical_control_point_positions.csv`
- Count distribution CSV: `outputs/luna16_saliency_control_point_distribution/empirical_nodule_count_distribution_from_control_points.csv`
- Sampling arrays NPZ: `outputs/luna16_saliency_control_point_distribution/empirical_nodule_distribution_from_control_points.npz`
- JSON summary: `outputs/luna16_saliency_control_point_distribution/empirical_nodule_distribution_from_control_points_summary.json`