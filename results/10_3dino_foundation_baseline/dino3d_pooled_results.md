# LUNA16 3DINO-ViT Partial-Fine-Tuning Baseline

3DINO-ViT (ViT-Large-3D, official AICONSlab checkpoint) with the first 12 of 24 transformer blocks frozen and the second 12 blocks + final norm fine-tuned, evaluated on the exact same 10 LUNA16 patient-level malignancy folds used by the Adaptive RBF, COLIPRI, and Rad-JEPA-3D experiments. No lesion-guided information (detections, nodule coordinates, masks) is used -- only overlapping 112^3 sliding windows over the full CT volume, aggregated via mean+max pooling.

## Pooled out-of-fold test metrics (all 10 folds concatenated)

| Representation | Folds | Samples | AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3dino_partial_ft | 10 | 796 | 0.5169 | 0.0338 | 0.5402 | 0.4097 | 0.4233 | 0.3969 | 303 | 173 | 193 | 127 |

## Per-fold mean +/- std (ddof=1) test metrics

| Representation | Folds | AUC | MCC | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3dino_partial_ft | 10 | 0.5545 +/- 0.0877 | -0.0004 +/- 0.0558 | 0.5415 +/- 0.0925 | 0.2622 +/- 0.2742 | 0.2636 +/- 0.2043 | 0.3738 +/- 0.4574 |
