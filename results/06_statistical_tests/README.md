# Statistical Tests For LUNA16 Patient-Level Classification

This folder contains paired statistical tests for the main pooled 10-fold LUNA16 results.

## Protocol

- Unit of analysis: patient/scan-level prediction.
- Evaluation set: pooled predictions from the 10 test folds, `n = 796` scans.
- Class distribution: `320` malignant, `476` benign.
- Main method: Adaptive RBF detector-guided synthetic 2D, `top-k=4`, `threshold=0.50`, EfficientNetV2-S.
- AUC comparison: paired DeLong test.
- Binary prediction comparison: exact McNemar/binomial test at the stored decision threshold.
- Confidence intervals: stratified patient-level bootstrap, `5000` resamples.
- Multiple testing correction: Benjamini-Hochberg FDR over the planned pairwise comparisons.

## Files

- `pooled_metrics_for_stat_tests.csv`: pooled metrics for every tested method.
- `bootstrap_95ci.csv`: 95% bootstrap confidence intervals.
- `paired_comparison_tests.csv`: paired DeLong and McNemar tests versus the main Adaptive RBF model.
- `statistical_test_protocol.json`: machine-readable protocol.
- `run_statistical_tests.py`: reproducible analysis script.

## Pooled Metrics

| Method | AUC | 95% CI AUC | MCC | 95% CI MCC | F1 |
|---|---:|---:|---:|---:|---:|
| Adaptive RBF, detector-guided | **0.8149** | 0.7843-0.8453 | **0.4780** | 0.4155-0.5402 | **0.6806** |
| Adaptive Shepard, detector-guided | 0.7793 | 0.7476-0.8109 | 0.4308 | 0.3671-0.4941 | 0.6283 |
| MIP tri-view | 0.6175 | 0.5786-0.6575 | 0.1556 | 0.0883-0.2240 | 0.5008 |
| Central axial slice | 0.6153 | 0.5743-0.6544 | 0.1610 | 0.0913-0.2284 | 0.4668 |
| Random-control RBF | 0.6105 | 0.5706-0.6492 | 0.1877 | 0.1184-0.2574 | 0.5207 |
| MIP axial | 0.6075 | 0.5663-0.6462 | 0.1604 | 0.0902-0.2301 | 0.5061 |
| Fixed-control RBF | 0.5878 | 0.5475-0.6263 | 0.1109 | 0.0401-0.1787 | 0.4499 |
| 3D ResNet18 fit-pad | 0.5659 | 0.5256-0.6050 | 0.0989 | 0.0297-0.1665 | 0.4870 |
| Detector crop-MIL attention | 0.5410 | 0.5294-0.5536 | 0.1861 | 0.1566-0.2145 | 0.5942 |

## Paired Comparisons Versus Adaptive RBF

| Comparator | Delta AUC | DeLong p | FDR p | Delta MCC | McNemar p | McNemar FDR p |
|---|---:|---:|---:|---:|---:|---:|
| Adaptive Shepard | +0.0356 | 2.55e-02 | 2.55e-02 | +0.0471 | 2.55e-01 | 2.55e-01 |
| MIP axial | +0.2074 | 2.05e-19 | 4.10e-19 | +0.3175 | 8.65e-14 | 1.73e-13 |
| MIP tri-view | +0.1974 | 9.09e-17 | 1.45e-16 | +0.3223 | 1.44e-12 | 2.31e-12 |
| Central axial slice | +0.1996 | 1.12e-16 | 1.50e-16 | +0.3170 | 6.33e-12 | 8.44e-12 |
| Detector crop-MIL attention | +0.2740 | 2.87e-62 | 2.30e-61 | +0.2919 | 9.42e-30 | 7.54e-29 |
| Fixed-control RBF | +0.2271 | 1.24e-20 | 3.31e-20 | +0.3670 | 2.87e-15 | 7.64e-15 |
| Random-control RBF | +0.2045 | 1.72e-16 | 1.96e-16 | +0.2903 | 1.66e-10 | 1.90e-10 |
| 3D ResNet18 fit-pad | +0.2491 | 1.06e-23 | 4.23e-23 | +0.3790 | 6.90e-18 | 2.76e-17 |

## Interpretation

The Adaptive RBF representation significantly improves AUC over all planned comparators after FDR correction. The strongest evidence is against non-adaptive volume-to-2D baselines, detector-crop MIL, fixed/random control-point ablations, and the 3D ResNet18 volumetric baseline.

The RBF-vs-Shepard comparison is also significant for AUC by paired DeLong test (`p = 0.0255`), supporting RBF as the preferred interpolation strategy. However, McNemar is not significant for this pair (`p = 0.2545`), meaning the binary error sets at the operating threshold are not clearly different. This is a useful nuance for the paper: RBF improves ranking/discrimination, while the thresholded classifications are closer.

The fixed-control and random-control ablations remain near the non-adaptive baselines. This supports the central claim that the performance gain is not due only to using an RBF surface or a 2D pretrained backbone, but to detector-guided placement of the control points.
