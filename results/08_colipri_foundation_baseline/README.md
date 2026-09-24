# COLIPRI-CRM Frozen Foundation-Model Baseline

Compares the Adaptive RBF approach against a **frozen** [COLIPRI-CRM](https://huggingface.co/microsoft/colipri) 3D CT
foundation-model encoder for LUNA16 patient-level benign-vs-malignant classification, using the exact same patient
labels and 10-fold splits as the rest of the project (`data/processed/cv_splits/luna16_classification_fold{0-9}.csv`).

Code: [`src/luna16_colipri_3d/`](../../src/luna16_colipri_3d/). Reproduce with:

```bash
CUDA_VISIBLE_DEVICES=3 bash bash/luna16_colipri_3d/run_colipri_baseline.sh
```

See the main [README](../../README.md#colipri-foundation-model-baseline) for a full description of the pipeline
(feature extraction with the official frozen encoder, linear probing, and cross-fold aggregation).

Two representations are compared, both 768-D:

- **pooled**: the official COLIPRI pooled+projected embedding.
- **dense_maxpool**: the dense 768x24x24x24 projected feature grid reduced with a channel-wise global max pool.

## Status

`complete` — run end-to-end on the full 888-patient cohort (796 binary benign/malignant patients across the 10
folds) on an NVIDIA A100 80GB. Full report: [`colipri_pooled_results.md`](colipri_pooled_results.md); raw tables:
[`pooled_metrics.csv`](pooled_metrics.csv), [`per_fold_mean_std_metrics.csv`](per_fold_mean_std_metrics.csv).

### Pooled out-of-fold test metrics (all 10 folds concatenated, 796 samples)

| Representation | AUC | MCC | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| pooled | 0.8508 | 0.5265 | 0.7676 | 0.7251 | 0.6912 | 0.7625 |
| dense_maxpool | 0.8264 | 0.5236 | 0.7663 | 0.7232 | 0.6903 | 0.7594 |

### Per-fold mean +/- std (ddof=1)

| Representation | AUC | MCC | Accuracy | F1 |
|---|---:|---:|---:|---:|
| pooled | 0.8476 +/- 0.0574 | 0.5248 +/- 0.1257 | 0.7679 +/- 0.0661 | 0.7235 +/- 0.0717 |
| dense_maxpool | 0.8462 +/- 0.0602 | 0.5288 +/- 0.1228 | 0.7665 +/- 0.0659 | 0.7234 +/- 0.0652 |

The official pooled embedding and the dense channel-max-pooled representation perform almost identically, with the
official pooled head slightly ahead on AUC. Compare against the Adaptive RBF pooled result (AUC 0.8149, MCC 0.4780;
[`results/00_paper_overview`](../00_paper_overview)) on the same patients/folds: the frozen COLIPRI-CRM linear
probe is a strong baseline, slightly exceeding the adaptive-surface pipeline on this split despite using no
task-specific training beyond a single linear layer.
