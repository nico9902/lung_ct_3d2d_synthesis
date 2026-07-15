# Empirical LUNA16 Nodule Position Distribution From Control Points

Source output root: `outputs/luna16_saliency_synthetic_gt`

This distribution is derived from `control_points_*.npy`, `point_labels_*.npy`, and preprocessed volume metadata when available.
If metadata is missing for a scan, the script falls back to the saved surface-grid extent for that scan.
Every 5 control points are treated as one nodule; the first point in each group is the nodule center used for the empirical position distribution.
It does **not** use annotation CSV coordinates.

## Summary

- Samples with control points: **601**
- Control points: **5895**
- Nodules: **1179**
- Positive-log filter enabled: **True**
- Positive scans in log: **601**
- Preprocessed root: `data/LUNA16_preprocessed`
- Missing metadata scans: **0**
- Skipped because not positive in log: **287**
- Skipped samples: **0**

## Relative Position Distribution

Nodule-center coordinates are normalized to each preprocessed volume size and saved in `[rel_z, rel_y, rel_x]`.

| Axis | Mean | Std | Q05 | Q25 | Q50 | Q75 | Q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rel_z | 0.515 | 0.197 | 0.210 | 0.359 | 0.505 | 0.667 | 0.845 |
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