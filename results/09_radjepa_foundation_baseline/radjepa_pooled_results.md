# LUNA16 Rad-JEPA-3D Frozen Linear-Probe Baseline

Frozen Rad-JEPA-3D (H-Mamba hybrid encoder) 3D foundation-model baseline with a trained `Linear(feature_dim, 1)` probe, evaluated on the exact same 10 LUNA16 patient-level malignancy folds used by the Adaptive RBF experiments. No lesion-guided information (detections, nodule coordinates, masks) is used -- Rad-JEPA sees the full CT volume only.

## Pooled out-of-fold test metrics (all 10 folds concatenated)

| Representation | Folds | Samples | AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resize32 | 10 | 796 | 0.5679 | 0.1373 | 0.5842 | 0.4868 | 0.4831 | 0.4906 | 308 | 168 | 163 | 157 |
| sliding_mean | 10 | 796 | 0.5838 | 0.1504 | 0.5917 | 0.4914 | 0.4922 | 0.4906 | 314 | 162 | 163 | 157 |
| sliding_max | 10 | 796 | 0.5765 | 0.1498 | 0.5829 | 0.5103 | 0.4832 | 0.5406 | 291 | 185 | 147 | 173 |
| sliding_mean_max | 10 | 796 | 0.5934 | 0.1615 | 0.5854 | 0.5231 | 0.4866 | 0.5656 | 285 | 191 | 139 | 181 |

## Per-fold mean +/- std (ddof=1) test metrics

| Representation | Folds | AUC | MCC | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| resize32 | 10 | 0.5779 +/- 0.0528 | 0.1232 +/- 0.1058 | 0.5844 +/- 0.0407 | 0.4697 +/- 0.1196 | 0.4686 +/- 0.0968 | 0.4815 +/- 0.1504 |
| sliding_mean | 10 | 0.5873 +/- 0.0584 | 0.1365 +/- 0.1155 | 0.5919 +/- 0.0464 | 0.4750 +/- 0.1221 | 0.4768 +/- 0.0975 | 0.4828 +/- 0.1472 |
| sliding_max | 10 | 0.6031 +/- 0.0555 | 0.1428 +/- 0.1203 | 0.5833 +/- 0.0550 | 0.4977 +/- 0.1111 | 0.4735 +/- 0.0810 | 0.5369 +/- 0.1491 |
| sliding_mean_max | 10 | 0.6078 +/- 0.0500 | 0.1619 +/- 0.1017 | 0.5858 +/- 0.0540 | 0.5209 +/- 0.0639 | 0.4869 +/- 0.0755 | 0.5668 +/- 0.0750 |
