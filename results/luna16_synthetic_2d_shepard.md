# LUNA16 Synthetic 2D Shepard Results

Sources:
- Ground truth Shepard: `outputs/luna16_synthetic_2d_gt_shepard/prediction_metrics.csv`
- Detector top5 minprob0.5 Shepard: `outputs/luna16_synthetic_2d_top5_minprob0.5_shepard/prediction_metrics.csv`

Rows below use `scope = pooled`, so each metric is recomputed after concatenating all test predictions available for the backbone across folds.

## Summary

- GT Shepard best pooled MCC: **efficientnet_v2_s** with **0.611**.
- GT Shepard best pooled AUC: **resnet50** with **0.856**.
- Detector Shepard best pooled MCC: **resnet18** with **0.338**.
- Detector Shepard best pooled AUC: **efficientnet_v2_s** with **0.729**.
- Pooled sample count per backbone: **796** total, **320** positive, **476** negative.
- Mean detector-minus-GT delta across backbones: MCC **-0.260**, AUC **-0.155**, F1 **-0.165**, accuracy **-0.122**.
- Detector Shepard is lower than GT Shepard for every backbone and every pooled metric in this run.

## GT Shepard Pooled Metrics

|   Rank | Backbone            |   Samples |   Positives |   Negatives |   Accuracy |    F1 |   MCC |   AUC |
|-------:|:--------------------|----------:|------------:|------------:|-----------:|------:|------:|------:|
|      1 | `efficientnet_v2_s` |       796 |         320 |         476 |      0.815 | 0.756 | 0.611 | 0.850 |
|      2 | `resnet50`          |       796 |         320 |         476 |      0.799 | 0.731 | 0.576 | 0.856 |
|      3 | `vgg16`             |       796 |         320 |         476 |      0.796 | 0.739 | 0.573 | 0.853 |
|      4 | `densenet121`       |       796 |         320 |         476 |      0.790 | 0.728 | 0.559 | 0.856 |
|      5 | `efficientnet_b0`   |       796 |         320 |         476 |      0.779 | 0.711 | 0.535 | 0.827 |
|      6 | `efficientnet_b1`   |       796 |         320 |         476 |      0.778 | 0.714 | 0.533 | 0.838 |
|      7 | `resnet18`          |       796 |         320 |         476 |      0.773 | 0.691 | 0.519 | 0.821 |

## Detector Top5 Minprob0.5 Shepard Pooled Metrics

|   Rank | Backbone            |   Samples |   Positives |   Negatives |   Accuracy |    F1 |   MCC |   AUC |
|-------:|:--------------------|----------:|------------:|------------:|-----------:|------:|------:|------:|
|      1 | `resnet18`          |       796 |         320 |         476 |      0.690 | 0.570 | 0.338 | 0.675 |
|      2 | `efficientnet_v2_s` |       796 |         320 |         476 |      0.682 | 0.596 | 0.335 | 0.729 |
|      3 | `vgg16`             |       796 |         320 |         476 |      0.687 | 0.567 | 0.332 | 0.700 |
|      4 | `efficientnet_b1`   |       796 |         320 |         476 |      0.667 | 0.558 | 0.294 | 0.684 |
|      5 | `densenet121`       |       796 |         320 |         476 |      0.661 | 0.530 | 0.274 | 0.680 |
|      6 | `efficientnet_b0`   |       796 |         320 |         476 |      0.653 | 0.555 | 0.272 | 0.674 |
|      7 | `resnet50`          |       796 |         320 |         476 |      0.638 | 0.541 | 0.243 | 0.671 |

## Detector Minus GT Comparison

Negative deltas mean detector-driven Shepard training is lower than ground-truth Shepard training.

| Backbone            |   GT MCC |   Detector MCC |   Delta MCC |   GT AUC |   Detector AUC |   Delta AUC |   GT F1 |   Detector F1 |   Delta F1 |   GT Acc |   Detector Acc |   Delta Acc |
|:--------------------|---------:|---------------:|------------:|---------:|---------------:|------------:|--------:|--------------:|-----------:|---------:|---------------:|------------:|
| `resnet18`          |    0.519 |          0.338 |      -0.181 |    0.821 |          0.675 |      -0.146 |   0.691 |         0.570 |     -0.120 |    0.773 |          0.690 |      -0.083 |
| `efficientnet_b1`   |    0.533 |          0.294 |      -0.239 |    0.838 |          0.684 |      -0.154 |   0.714 |         0.558 |     -0.156 |    0.778 |          0.667 |      -0.111 |
| `vgg16`             |    0.573 |          0.332 |      -0.241 |    0.853 |          0.700 |      -0.153 |   0.739 |         0.567 |     -0.172 |    0.796 |          0.687 |      -0.109 |
| `efficientnet_b0`   |    0.535 |          0.272 |      -0.263 |    0.827 |          0.674 |      -0.153 |   0.711 |         0.555 |     -0.157 |    0.779 |          0.653 |      -0.126 |
| `efficientnet_v2_s` |    0.611 |          0.335 |      -0.277 |    0.850 |          0.729 |      -0.121 |   0.756 |         0.596 |     -0.160 |    0.815 |          0.682 |      -0.133 |
| `densenet121`       |    0.559 |          0.274 |      -0.285 |    0.856 |          0.680 |      -0.176 |   0.728 |         0.530 |     -0.199 |    0.790 |          0.661 |      -0.129 |
| `resnet50`          |    0.576 |          0.243 |      -0.333 |    0.856 |          0.671 |      -0.185 |   0.731 |         0.541 |     -0.189 |    0.799 |          0.638 |      -0.161 |

## Cross-Fold Mean And Std

These values summarize the 10 individual fold rows per backbone, rather than the pooled row.

| Source           | Backbone            |   Acc mean |   Acc std |   F1 mean |   F1 std |   MCC mean |   MCC std |   AUC mean |   AUC std |
|:-----------------|:--------------------|-----------:|----------:|----------:|---------:|-----------:|----------:|-----------:|----------:|
| Detector Shepard | `efficientnet_v2_s` |      0.682 |     0.041 |     0.573 |    0.138 |      0.331 |     0.108 |      0.730 |     0.104 |
| Detector Shepard | `vgg16`             |      0.687 |     0.050 |     0.554 |    0.115 |      0.323 |     0.130 |      0.720 |     0.063 |
| Detector Shepard | `efficientnet_b0`   |      0.653 |     0.041 |     0.541 |    0.119 |      0.273 |     0.124 |      0.701 |     0.060 |
| Detector Shepard | `efficientnet_b1`   |      0.667 |     0.039 |     0.531 |    0.144 |      0.277 |     0.106 |      0.701 |     0.066 |
| Detector Shepard | `densenet121`       |      0.660 |     0.061 |     0.501 |    0.162 |      0.263 |     0.129 |      0.693 |     0.057 |
| Detector Shepard | `resnet50`          |      0.638 |     0.042 |     0.521 |    0.124 |      0.239 |     0.092 |      0.674 |     0.045 |
| Detector Shepard | `resnet18`          |      0.689 |     0.041 |     0.539 |    0.172 |      0.332 |     0.100 |      0.666 |     0.083 |
| GT Shepard       | `vgg16`             |      0.797 |     0.034 |     0.738 |    0.027 |      0.581 |     0.062 |      0.878 |     0.022 |
| GT Shepard       | `efficientnet_v2_s` |      0.815 |     0.048 |     0.756 |    0.059 |      0.616 |     0.090 |      0.865 |     0.053 |
| GT Shepard       | `densenet121`       |      0.791 |     0.047 |     0.730 |    0.043 |      0.562 |     0.082 |      0.857 |     0.057 |
| GT Shepard       | `resnet50`          |      0.800 |     0.057 |     0.730 |    0.078 |      0.578 |     0.113 |      0.850 |     0.053 |
| GT Shepard       | `efficientnet_b1`   |      0.778 |     0.044 |     0.711 |    0.058 |      0.534 |     0.086 |      0.840 |     0.044 |
| GT Shepard       | `efficientnet_b0`   |      0.779 |     0.051 |     0.712 |    0.058 |      0.536 |     0.101 |      0.839 |     0.038 |
| GT Shepard       | `resnet18`          |      0.773 |     0.046 |     0.687 |    0.061 |      0.520 |     0.083 |      0.836 |     0.042 |

## Notes

- The detector Shepard results are consistent with the QC report in `docs/luna16_saliency_synthetic_shepard_qc_report`: detector surfaces have higher geometric roughness and larger `z_range`, which likely contributes to the lower classification metrics.
- `resnet18` is the detector Shepard winner by pooled MCC, but `efficientnet_v2_s` is the detector Shepard winner by pooled AUC.
- GT Shepard remains much stronger than detector Shepard, with the smallest MCC gap for `resnet18` and the largest MCC gap for `resnet50`.
