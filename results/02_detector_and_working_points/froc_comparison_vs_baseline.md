# FROC Comparison vs Baseline Detector

Current detector:
`outputs/scpmnet_luna16_10fold_normauto_with_fold8_retry`

Baseline detector:
`outputs/scpmnet_luna16_10fold`

Comparison source files:

- Current: `cv_aggregate/pooled_test_froc.csv`
- Baseline: `outputs/scpmnet_luna16_10fold/cv_aggregate/pooled_test_froc.csv`

| FP/scan | baseline sensitivity | current sensitivity | delta |
|---:|---:|---:|---:|
| 0.125 | 0.000 | 0.365 | +0.365 |
| 0.25 | 0.017 | 0.476 | +0.459 |
| 0.5 | 0.346 | 0.589 | +0.243 |
| 1.0 | 0.576 | 0.701 | +0.125 |
| 2.0 | 0.705 | 0.781 | +0.076 |
| 4.0 | 0.775 | 0.834 | +0.059 |
| 8.0 | 0.825 | 0.874 | +0.049 |

Mean pooled FROC:

- Baseline: `0.463`
- Current detector: `0.660`
- Absolute improvement: `+0.196`
- Relative improvement: `+42.4%`

The largest gains are at low FP/scan, especially `0.125` and `0.25` FP/scan.

## Excluding Fold8

To separate the effect of the fold8 retry from the effect of `normauto`, the table below recomputes the cross-validation average after removing fold8 from both detectors. This is a fold-level average over folds 0-7 and 9.

| metric | baseline without fold8 | current without fold8 | delta |
|---|---:|---:|---:|
| mean FROC | 0.662 | 0.681 | +0.019 |
| FROC 0.125 FP/scan | 0.381 | 0.396 | +0.015 |
| FROC 0.25 FP/scan | 0.493 | 0.504 | +0.011 |
| FROC 0.5 FP/scan | 0.595 | 0.605 | +0.009 |
| FROC 1 FP/scan | 0.693 | 0.717 | +0.024 |
| FROC 2 FP/scan | 0.775 | 0.804 | +0.029 |
| FROC 4 FP/scan | 0.831 | 0.848 | +0.017 |
| FROC 8 FP/scan | 0.864 | 0.892 | +0.029 |

Fold8 alone:

| metric | baseline fold8 | current fold8 | delta |
|---|---:|---:|---:|
| mean FROC | 0.151 | 0.587 | +0.436 |
| FROC 0.125 FP/scan | 0.000 | 0.407 | +0.407 |
| FROC 0.25 FP/scan | 0.000 | 0.449 | +0.449 |
| FROC 0.5 FP/scan | 0.000 | 0.551 | +0.551 |
| FROC 1 FP/scan | 0.000 | 0.602 | +0.602 |
| FROC 2 FP/scan | 0.000 | 0.653 | +0.653 |
| FROC 4 FP/scan | 0.314 | 0.695 | +0.381 |
| FROC 8 FP/scan | 0.746 | 0.754 | +0.008 |

Interpretation: without fold8, the detector still improves over the baseline, but only slightly (`+0.019` mean FROC). Most of the full 10-fold gain comes from fixing fold8, where mean FROC improves by `+0.436`.
