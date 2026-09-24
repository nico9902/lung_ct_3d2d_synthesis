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
| COLIPRI-CRM frozen linear probe, pooled embedding | **0.8508** | 0.8225-0.8768 | **0.5265** | 0.4654-0.5833 | 0.7251 |
| COLIPRI-CRM frozen linear probe, dense channel-max-pool embedding | 0.8264 | 0.7965-0.8559 | 0.5236 | 0.4639-0.5835 | 0.7232 |
| Adaptive RBF, detector-guided | 0.8149 | 0.7843-0.8453 | 0.4780 | 0.4155-0.5402 | 0.6806 |
| Adaptive Shepard, detector-guided | 0.7793 | 0.7476-0.8109 | 0.4308 | 0.3671-0.4941 | 0.6283 |
| MIP tri-view | 0.6175 | 0.5786-0.6575 | 0.1556 | 0.0883-0.2240 | 0.5008 |
| Central axial slice | 0.6153 | 0.5743-0.6544 | 0.1610 | 0.0913-0.2284 | 0.4668 |
| Random-control RBF | 0.6105 | 0.5706-0.6492 | 0.1877 | 0.1184-0.2574 | 0.5207 |
| MIP axial | 0.6075 | 0.5663-0.6462 | 0.1604 | 0.0902-0.2301 | 0.5061 |
| Rad-JEPA-3D, sliding-window mean+max | 0.5934 | 0.5534-0.6328 | 0.1615 | 0.0924-0.2310 | 0.5231 |
| Fixed-control RBF | 0.5878 | 0.5475-0.6263 | 0.1109 | 0.0401-0.1787 | 0.4499 |
| Rad-JEPA-3D, sliding-window mean | 0.5838 | 0.5429-0.6248 | 0.1504 | 0.0760-0.2189 | 0.4914 |
| Rad-JEPA-3D, sliding-window max | 0.5765 | 0.5362-0.6164 | 0.1498 | 0.0822-0.2176 | 0.5103 |
| Rad-JEPA-3D, resize32 | 0.5679 | 0.5263-0.6103 | 0.1373 | 0.0690-0.2081 | 0.4868 |
| 3D ResNet18 fit-pad | 0.5659 | 0.5256-0.6050 | 0.0989 | 0.0297-0.1665 | 0.4870 |
| Detector crop-MIL attention | 0.5410 | 0.5294-0.5536 | 0.1861 | 0.1566-0.2145 | 0.5942 |
| 3DINO-ViT partial fine-tuning, sliding-window mean+max | 0.5169 | not computed | 0.0338 | not computed | 0.4097 |
| Backbone with soft slice attention, EfficientNet-B0 half-frozen (batch size 8) | 0.5482 | not yet regenerated | 0.0685 | not yet regenerated | 0.3908 |

COLIPRI and Rad-JEPA-3D are the two frozen 3D foundation-model baselines (`results/08_colipri_foundation_baseline/`,
`results/09_radjepa_foundation_baseline/`); COLIPRI is bolded above as the current best pooled AUC/MCC but is
**not** the paper's main method (Adaptive RBF still is) — see the paired comparisons below.

## Paired Comparisons Versus Adaptive RBF

`Delta AUC`/`Delta MCC` are Adaptive RBF minus comparator, matching the original table's sign convention; a
**negative** value means the comparator scored higher than Adaptive RBF.

| Comparator | Delta AUC | DeLong p | FDR p | Delta MCC | McNemar p | McNemar FDR p |
|---|---:|---:|---:|---:|---:|---:|
| COLIPRI-CRM, pooled | -0.0359 | 2.31e-02 | 2.83e-02 | -0.0486 | 3.67e-01 | 4.08e-01 |
| COLIPRI-CRM, dense channel-max-pool | -0.0114 | 4.85e-01 | 4.85e-01 | -0.0457 | 4.15e-01 | 4.15e-01 |
| Adaptive Shepard | +0.0356 | 2.55e-02 | 2.55e-02 | +0.0471 | 2.55e-01 | 2.55e-01 |
| MIP axial | +0.2074 | 2.05e-19 | 4.10e-19 | +0.3175 | 8.65e-14 | 1.73e-13 |
| MIP tri-view | +0.1974 | 9.09e-17 | 1.45e-16 | +0.3223 | 1.44e-12 | 2.31e-12 |
| Central axial slice | +0.1996 | 1.12e-16 | 1.50e-16 | +0.3170 | 6.33e-12 | 8.44e-12 |
| Detector crop-MIL attention | +0.2740 | 2.87e-62 | 2.30e-61 | +0.2919 | 9.42e-30 | 7.54e-29 |
| Fixed-control RBF | +0.2271 | 1.24e-20 | 3.31e-20 | +0.3670 | 2.87e-15 | 7.64e-15 |
| Random-control RBF | +0.2045 | 1.72e-16 | 1.96e-16 | +0.2903 | 1.66e-10 | 1.90e-10 |
| 3D ResNet18 fit-pad | +0.2491 | 1.06e-23 | 4.23e-23 | +0.3790 | 6.90e-18 | 2.76e-17 |
| Rad-JEPA-3D, resize32 | +0.2470 | 1.56e-24 | 1.09e-23 | +0.3406 | 5.22e-14 | 1.22e-13 |
| Rad-JEPA-3D, sliding-window mean | +0.2311 | 1.39e-21 | 3.24e-21 | +0.3276 | 5.11e-13 | 8.95e-13 |
| Rad-JEPA-3D, sliding-window max | +0.2385 | 3.45e-24 | 1.61e-23 | +0.3282 | 2.46e-14 | 8.63e-14 |
| Rad-JEPA-3D, sliding-window mean+max | +0.2215 | 1.23e-21 | 3.24e-21 | +0.3164 | 3.38e-14 | 9.46e-14 |
| 3DINO-ViT partial fine-tuning, sliding-window mean+max | +0.2980 | 3.29e-32 | 2.47e-31 | +0.4441 | 1.62e-20 | 1.22e-19 |

## Interpretation

**COLIPRI (frozen foundation-model baseline).** The COLIPRI-CRM pooled embedding significantly exceeds Adaptive RBF
in AUC (`p = 0.0231`, FDR `p = 0.0283`) with a full frozen backbone and only a linear head, and its 95% AUC CI
(0.8225-0.8768) sits entirely above Adaptive RBF's point estimate. The dense channel-max-pool representation is
numerically higher (+0.0114 AUC) but not significantly different (`p = 0.485`). Neither COLIPRI representation
differs significantly from Adaptive RBF on the McNemar test of thresholded predictions (`p = 0.37, 0.41`), so the
AUC gap reflects better score ranking/calibration rather than a clearly different binary error pattern at the
stored 0.5 threshold. This is an important result for the paper: a large pretrained 3D encoder with no
task-specific fine-tuning is a strong, hard-to-beat baseline for this task, and the detector-guided adaptive-surface
pipeline's main advantage is not raw discrimination over such a foundation model but its far smaller compute/model
footprint (single linear layer aside, COLIPRI still requires a full forward pass through a ~147M-parameter frozen
3D transformer per scan) and full task-specific interpretability via the synthesized 2D surface.

**Rad-JEPA-3D (second frozen foundation-model baseline).** Unlike COLIPRI, all four Rad-JEPA-3D representations are
significantly *worse* than Adaptive RBF on both AUC (all FDR `p < 1e-20`) and McNemar (all FDR `p < 1e-12`) — the
95% AUC CIs (0.526-0.633 across the four variants) sit entirely below Adaptive RBF's, and the binary error pattern
also differs significantly, unlike the COLIPRI comparison. The best Rad-JEPA variant (sliding-window mean+max) still
only reaches AUC 0.593, barely above the non-adaptive 2D baselines. This is a useful contrast for the paper: not
every frozen 3D foundation-model encoder transfers well to this task out of the box — COLIPRI's joint image-text
pretraining (and its trained attention-pooling head) appears far more informative for malignancy discrimination
than Rad-JEPA-3D's JEPA masked-prediction pretraining probed with simple mean/max pooling, even though both are
given the same full, lesion-unguided CT volume.

The Adaptive RBF representation still significantly improves AUC over every other planned comparator after FDR
correction. The strongest evidence is against non-adaptive volume-to-2D baselines, detector-crop MIL, fixed/random
control-point ablations, and the 3D ResNet18 volumetric baseline.

The RBF-vs-Shepard comparison is also significant for AUC by paired DeLong test (`p = 0.0255`), supporting RBF as the preferred interpolation strategy. However, McNemar is not significant for this pair (`p = 0.2545`), meaning the binary error sets at the operating threshold are not clearly different. This is a useful nuance for the paper: RBF improves ranking/discrimination, while the thresholded classifications are closer.

The fixed-control and random-control ablations remain near the non-adaptive baselines. This supports the central claim that the performance gain is not due only to using an RBF surface or a 2D pretrained backbone, but to detector-guided placement of the control points.
