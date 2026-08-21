# CPMNetv2 LUNA16 10-fold Results

Run: `20260809_cpmnetv2_luna16_10fold_bs8_numsam3_topk7_a100_lrbase0001_lrmax001_bf16_guarded`

Coverage grid: `topk=1..10`, `threshold=0.00..0.50` step `0.05`.

## Pooled Test FROC

Questa e la FROC aggregata vera: prediction e GT test dei 10 fold concatenate e valutate come un unico dataset.

| FP/scan | Sensitivity |
| --- | ---: |
| 0.125 | 0.499157 |
| 0.25 | 0.644182 |
| 0.5 | 0.764755 |
| 1 | 0.851602 |
| 2 | 0.920742 |
| 4 | 0.954469 |
| 8 | 0.977234 |

Pooled mean FROC: **0.801735**

## Fold-Average Test FROC

Questa sezione riporta invece media e deviazione standard delle metriche calcolate separatamente sui 10 fold.


| FP/scan | Sensitivity mean | Std |
| --- | ---: | ---: |
| 0.125 | 0.5614 | 0.0983 |
| 0.25 | 0.6664 | 0.0938 |
| 0.5 | 0.7656 | 0.0697 |
| 1 | 0.8605 | 0.0438 |
| 2 | 0.9213 | 0.0283 |
| 4 | 0.9610 | 0.0244 |
| 8 | 0.9791 | 0.0145 |

Mean FROC: **0.8165 ± 0.0497**

## Fold Mean FROC

| Fold | Mean FROC |
| ---: | ---: |
| 0 | 0.8729 |
| 1 | 0.9049 |
| 2 | 0.8356 |
| 3 | 0.7313 |
| 4 | 0.7870 |
| 5 | 0.7892 |
| 6 | 0.8652 |
| 7 | 0.7810 |
| 8 | 0.8085 |
| 9 | 0.7893 |

## Selected Coverage Points

| Threshold | Top-k | Nodule coverage | FP/scan | Detections kept |
| ---: | ---: | ---: | ---: | ---: |
| 0.45 | 7 | 92.07% | 3.718 | 4394 |
| 0.45 | 8 | 93.42% | 4.062 | 4715 |
| 0.45 | 10 | 95.36% | 4.595 | 5211 |
| 0.50 | 7 | 91.57% | 3.145 | 3879 |
| 0.50 | 8 | 92.92% | 3.387 | 4110 |
| 0.50 | 10 | 94.60% | 3.743 | 4446 |

## Best Coverage Rows

| Threshold | Top-k | Nodule coverage | Positive-scan coverage | FP/scan |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 10 | 95.78% | 99.17% | 8.276 |
| 0.20 | 10 | 95.78% | 99.17% | 8.646 |
| 0.00 | 10 | 95.78% | 99.17% | 8.717 |
| 0.05 | 10 | 95.78% | 99.17% | 8.717 |
| 0.10 | 10 | 95.78% | 99.17% | 8.717 |
| 0.15 | 10 | 95.78% | 99.17% | 8.717 |
| 0.40 | 10 | 95.70% | 99.17% | 5.613 |
| 0.35 | 10 | 95.70% | 99.17% | 6.605 |
| 0.30 | 10 | 95.70% | 99.17% | 7.599 |
| 0.45 | 10 | 95.36% | 99.00% | 4.595 |

## Files

- `fold_froc_summary.csv`
- `aggregate_froc_summary.csv`
- `coverage_topk_threshold/coverage_by_threshold_topk.csv`
- `coverage_topk_threshold/pooled_coverage.csv`
- `coverage_topk_threshold/nodule_coverage_matrix_percent.csv`
- `coverage_topk_threshold/false_positives_per_scan_matrix.csv`
- `coverage_topk_threshold/detections_kept_matrix.csv`
- `coverage_topk_threshold/assets/nodule_coverage_heatmap.svg`
- `coverage_topk_threshold/assets/positive_scan_coverage_heatmap.svg`

- `cv_aggregate_pooled/pooled_test_froc.csv`
- `cv_aggregate_pooled/pooled_test_froc_curve.csv`
- `cv_aggregate_pooled/pooled_test_predictions.csv`
