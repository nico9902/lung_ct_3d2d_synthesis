# LUNA16 2.5D Slice-Attention Baseline

Non-adaptive patient-level classifier over all axial slices from the preprocessed LUNA16 volume. This completed 10-fold run uses an ImageNet-pretrained EfficientNet-B0 with approximately half of the backbone frozen, batch size 8, and one accumulation step; it is therefore reported separately from the planned fully fine-tuned EfficientNetV2-S experiment.

| Backbone | Folds | Samples | AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| efficientnet_b0 | 10 | 796 | 0.5482 | 0.0685 | 0.5691 | 0.3908 | 0.4527 | 0.3438 | 343 | 133 | 210 | 110 |

Interpretation: this baseline tests whether a pretrained 2D encoder plus learned soft attention over native axial context can match the adaptive detector-guided 2D surface. Predictions are stored under `outputs/luna16_slice_attention_2p5d_effnetb0_half_frozen_v100_bs2/all_test_predictions_efficientnet_b0.csv`.
