# LUNA16 COLIPRI-CRM Frozen Linear-Probe Baseline

Frozen COLIPRI-CRM 3D foundation-model encoder (official checkpoint, official 192x192x192 @ 2mm-isotropic preprocessing) with a trained `Linear(768, 1)` probe, evaluated on the exact same 10 LUNA16 patient-level malignancy folds used by the Adaptive RBF experiments.

## Pooled out-of-fold test metrics (all 10 folds concatenated)

| Representation | Folds | Samples | AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled | 10 | 796 | 0.8508 | 0.5265 | 0.7676 | 0.7251 | 0.6912 | 0.7625 | 367 | 109 | 76 | 244 |
| dense_maxpool | 10 | 796 | 0.8264 | 0.5236 | 0.7663 | 0.7232 | 0.6903 | 0.7594 | 367 | 109 | 77 | 243 |

## Per-fold mean +/- std (ddof=1) test metrics

| Representation | Folds | AUC | MCC | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| pooled | 10 | 0.8476 +/- 0.0574 | 0.5248 +/- 0.1257 | 0.7679 +/- 0.0661 | 0.7235 +/- 0.0717 | 0.6924 +/- 0.0802 | 0.7613 +/- 0.0768 |
| dense_maxpool | 10 | 0.8462 +/- 0.0602 | 0.5288 +/- 0.1228 | 0.7665 +/- 0.0659 | 0.7234 +/- 0.0652 | 0.6958 +/- 0.0852 | 0.7643 +/- 0.0906 |
