# LUNA16 Detector Top7 MinProb 0.3 Synthetic QC Comparison

This report summarizes QC for the detector-driven synthetic images generated with:

- `data/luna16_saliency_synthetic_detector_top7_minprob0.3_rbf`
- `data/luna16_saliency_synthetic_detector_top7_minprob0.3_shepard`

Lower `qc_badness` is better. Buckets follow the existing project heuristic:
`good_or_low_review <= 0.10`, `moderate_review <= 0.30`, `high_review > 0.30`.

## Summary

| method | n | qc_mean | qc_median | qc_p95 | good/low | moderate | high | z_range_mean | grad_p99_mean | better_than_gt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RBF top7 minprob0.3 | 888 | 0.389 | 0.379 | 0.641 | 18 | 209 | 661 | 148.1 | 3.503 | 62 / 888 (7.0%) |
| Shepard top7 minprob0.3 | 888 | 0.344 | 0.346 | 0.545 | 20 | 300 | 568 | 137.0 | 3.161 | 36 / 888 (4.1%) |

## Baseline Context

Compared with the previous top5/minprob0.5 detector sets:

| method | qc_mean | qc_median | qc_p95 | high_review | z_range_mean | grad_p99_mean |
|---|---:|---:|---:|---:|---:|---:|
| RBF top5 minprob0.5 | 0.179 | 0.147 | 0.477 | 204 | 67.5 | 1.352 |
| Shepard top5 minprob0.5 | 0.154 | 0.131 | 0.433 | 161 | 61.0 | 1.189 |
| RBF top7 minprob0.3 | 0.389 | 0.379 | 0.641 | 661 | 148.1 | 3.503 |
| Shepard top7 minprob0.3 | 0.344 | 0.346 | 0.545 | 568 | 137.0 | 3.161 |

## Interpretation

Top7 with threshold 0.3 is substantially noisier than top5 with threshold 0.5. In the generated top7 sets, 825 / 888 scans have 35 detector control labels, while the older top5/minprob0.5 sets are spread across fewer control-label counts. This increases cranio-caudal span and surface roughness, visible in both `z_range_mean` and `grad_p99_mean`.

Shepard is slightly cleaner than RBF for this top7/minprob0.3 setting: lower mean and p95 `qc_badness`, fewer high-review cases, and lower geometric roughness. However, both top7 sets are much worse than the existing top5/minprob0.5 sets by this QC heuristic.

## Generated Files

- `docs/luna16_saliency_synthetic_detector_top7_minprob03_rbf_qc_report/`
- `docs/luna16_saliency_synthetic_detector_top7_minprob03_shepard_qc_report/`
- `ground_truth_surface_qc_metrics.csv`
- `detector_top5_surface_qc_metrics.csv` contains the detector metrics for the requested top7 run; the filename is inherited from the existing QC script.
- `paired_surface_qc_comparison.csv`
