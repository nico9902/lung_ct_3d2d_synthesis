# LUNA16 Detection Coverage Map

Prediction root: `outputs/scpmnet_luna16_10fold_fpr_top100_focal_balanced_average`
Prediction file: `test_predictions_rescored.csv`
Score column: `probability`

A GT nodule is covered when one selected detection center falls inside the GT nodule radius. Matching is one-to-one.

![Nodule coverage heatmap](assets/nodule_coverage_heatmap.svg)

## Best Pooled Coverage Rows

| threshold | topk | covered_nodules | total_nodules | nodule_coverage | positive_scan_coverage | false_positives_per_scan | detections_kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 10.0000 | 1005.0000 | 1186.0000 | 0.8474 | 0.9501 | 8.8682 | 8880.0000 |
| 0.0500 | 10.0000 | 1005.0000 | 1186.0000 | 0.8474 | 0.9501 | 8.8682 | 8880.0000 |
| 0.1000 | 10.0000 | 1005.0000 | 1186.0000 | 0.8474 | 0.9501 | 8.8682 | 8880.0000 |
| 0.1500 | 10.0000 | 1005.0000 | 1186.0000 | 0.8474 | 0.9501 | 8.8682 | 8880.0000 |
| 0.2000 | 10.0000 | 1005.0000 | 1186.0000 | 0.8474 | 0.9501 | 8.8682 | 8880.0000 |
| 0.2500 | 10.0000 | 1005.0000 | 1186.0000 | 0.8474 | 0.9501 | 8.8682 | 8880.0000 |
| 0.3000 | 10.0000 | 1003.0000 | 1186.0000 | 0.8457 | 0.9501 | 8.4730 | 8527.0000 |
| 0.0000 | 9.0000 | 991.0000 | 1186.0000 | 0.8356 | 0.9484 | 7.8840 | 7992.0000 |
| 0.0500 | 9.0000 | 991.0000 | 1186.0000 | 0.8356 | 0.9484 | 7.8840 | 7992.0000 |
| 0.1000 | 9.0000 | 991.0000 | 1186.0000 | 0.8356 | 0.9484 | 7.8840 | 7992.0000 |

## Efficient Rows With >=80% Nodule Coverage

| threshold | topk | covered_nodules | total_nodules | nodule_coverage | positive_scan_coverage | false_positives_per_scan | detections_kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3500 | 8.0000 | 953.0000 | 1186.0000 | 0.8035 | 0.9235 | 5.4054 | 5753.0000 |
| 0.3000 | 7.0000 | 957.0000 | 1186.0000 | 0.8069 | 0.9451 | 5.7601 | 6072.0000 |
| 0.0000 | 7.0000 | 958.0000 | 1186.0000 | 0.8078 | 0.9451 | 5.9212 | 6216.0000 |
| 0.0500 | 7.0000 | 958.0000 | 1186.0000 | 0.8078 | 0.9451 | 5.9212 | 6216.0000 |
| 0.1000 | 7.0000 | 958.0000 | 1186.0000 | 0.8078 | 0.9451 | 5.9212 | 6216.0000 |
| 0.1500 | 7.0000 | 958.0000 | 1186.0000 | 0.8078 | 0.9451 | 5.9212 | 6216.0000 |
| 0.2000 | 7.0000 | 958.0000 | 1186.0000 | 0.8078 | 0.9451 | 5.9212 | 6216.0000 |
| 0.2500 | 7.0000 | 958.0000 | 1186.0000 | 0.8078 | 0.9451 | 5.9212 | 6216.0000 |
| 0.3500 | 9.0000 | 963.0000 | 1186.0000 | 0.8120 | 0.9251 | 6.0045 | 6295.0000 |
| 0.3500 | 10.0000 | 977.0000 | 1186.0000 | 0.8238 | 0.9268 | 6.5541 | 6797.0000 |

## Files

- `coverage_by_threshold_topk.csv`: fold and pooled coverage for every threshold/top-k pair.
- `pooled_coverage.csv`: pooled rows only.
- `missed_nodules.csv`: missed GT nodules for every threshold/top-k pair.
- `assets/nodule_coverage_heatmap.svg`: pooled nodule coverage map.
- `assets/positive_scan_coverage_heatmap.svg`: pooled positive-scan coverage map.
