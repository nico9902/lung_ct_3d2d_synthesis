# LUNA16 Synthetic 2D Pooled Metrics

Sources:
- Ground truth: `outputs/luna16_synthetic_2d_gt/prediction_metrics.csv`
- Detector top5 minprob0.5: `outputs/luna16_synthetic_2d_top5_minprob0.5/prediction_metrics.csv`
- Ground truth, half backbone frozen: `outputs/luna16_synthetic_2d_half_backbone_gt/prediction_metrics.csv`
- Detector top5 minprob0.5, half backbone frozen: `outputs/luna16_synthetic_2d_half_backbone_top5_minprob0.5/prediction_metrics.csv`

Rows below use `scope = pooled`, so each metric is recomputed after concatenating all test predictions available for the backbone across folds.

## Summary

- GT best pooled MCC: **vgg16** with **0.616**.
- Top5/minprob0.5 best pooled MCC: **efficientnet_v2_s** with **0.409**.
- GT best pooled AUC: **vgg16** with **0.865**.
- Top5/minprob0.5 best pooled AUC: **efficientnet_v2_s** with **0.741**.
- GT half-backbone-frozen best pooled MCC/AUC: **densenet121** with **0.412** MCC and **0.775** AUC.
- Top5/minprob0.5 half-backbone-frozen best pooled MCC/AUC: **densenet121** with **0.242** MCC and **0.648** AUC.
- Pooled sample count per backbone: **796** total, **320** positive, **476** negative.
- Mean top5-minus-GT delta across backbones: MCC **-0.247**, AUC **-0.137**, F1 **-0.157**, accuracy **-0.116**.
- Mean top5-minus-GT delta across backbones when freezing half the backbone: MCC **-0.129**, AUC **-0.082**, F1 **-0.093**, accuracy **-0.057**.

## GT Pooled Metrics

|   Rank | Backbone            |   Samples |   Positives |   Negatives |   Accuracy |    F1 |   MCC |   AUC |
|-------:|:--------------------|----------:|------------:|------------:|-----------:|------:|------:|------:|
|      1 | `vgg16`             |       796 |         320 |         476 |      0.817 | 0.765 | 0.616 | 0.865 |
|      2 | `efficientnet_v2_s` |       796 |         320 |         476 |      0.813 | 0.759 | 0.607 | 0.855 |
|      3 | `efficientnet_b0`   |       796 |         320 |         476 |      0.795 | 0.727 | 0.568 | 0.829 |
|      4 | `resnet50`          |       796 |         320 |         476 |      0.786 | 0.723 | 0.551 | 0.853 |
|      5 | `efficientnet_b1`   |       796 |         320 |         476 |      0.781 | 0.714 | 0.540 | 0.817 |
|      6 | `densenet121`       |       796 |         320 |         476 |      0.779 | 0.712 | 0.535 | 0.843 |
|      7 | `resnet18`          |       796 |         320 |         476 |      0.765 | 0.690 | 0.504 | 0.829 |

## Detector Top5 Minprob0.5 Pooled Metrics

|   Rank | Backbone            |   Samples |   Positives |   Negatives |   Accuracy |    F1 |   MCC |   AUC |
|-------:|:--------------------|----------:|------------:|------------:|-----------:|------:|------:|------:|
|      1 | `efficientnet_v2_s` |       796 |         320 |         476 |      0.722 | 0.614 | 0.409 | 0.741 |
|      2 | `efficientnet_b0`   |       796 |         320 |         476 |      0.692 | 0.574 | 0.343 | 0.728 |
|      3 | `resnet50`          |       796 |         320 |         476 |      0.682 | 0.570 | 0.324 | 0.702 |
|      4 | `vgg16`             |       796 |         320 |         476 |      0.668 | 0.573 | 0.303 | 0.688 |
|      5 | `densenet121`       |       796 |         320 |         476 |      0.667 | 0.565 | 0.297 | 0.689 |
|      6 | `efficientnet_b1`   |       796 |         320 |         476 |      0.653 | 0.534 | 0.263 | 0.695 |
|      7 | `resnet18`          |       796 |         320 |         476 |      0.637 | 0.561 | 0.252 | 0.689 |

## GT Half-Backbone-Frozen Pooled Metrics

|   Rank | Backbone            |   Samples |   Positives |   Negatives |   Accuracy |    F1 |   MCC |   AUC |
|-------:|:--------------------|----------:|------------:|------------:|-----------:|------:|------:|------:|
|      1 | `densenet121`       |       796 |         320 |         476 |      0.721 | 0.635 | 0.412 | 0.775 |
|      2 | `resnet50`          |       796 |         320 |         476 |      0.707 | 0.607 | 0.379 | 0.731 |
|      3 | `efficientnet_b1`   |       796 |         320 |         476 |      0.692 | 0.603 | 0.353 | 0.739 |
|      4 | `efficientnet_v2_s` |       796 |         320 |         476 |      0.693 | 0.560 | 0.343 | 0.711 |
|      5 | `efficientnet_b0`   |       796 |         320 |         476 |      0.681 | 0.584 | 0.327 | 0.711 |
|      6 | `resnet18`          |       796 |         320 |         476 |      0.637 | 0.559 | 0.251 | 0.666 |
|      7 | `vgg16`             |       796 |         320 |         476 |      0.633 | 0.535 | 0.232 | 0.663 |

## Detector Top5 Minprob0.5 Half-Backbone-Frozen Pooled Metrics

|   Rank | Backbone            |   Samples |   Positives |   Negatives |   Accuracy |    F1 |   MCC |   AUC |
|-------:|:--------------------|----------:|------------:|------------:|-----------:|------:|------:|------:|
|      1 | `densenet121`       |       796 |         320 |         476 |      0.644 | 0.518 | 0.242 | 0.648 |
|      2 | `efficientnet_v2_s` |       796 |         320 |         476 |      0.631 | 0.484 | 0.207 | 0.623 |
|      3 | `efficientnet_b0`   |       796 |         320 |         476 |      0.626 | 0.503 | 0.206 | 0.646 |
|      4 | `efficientnet_b1`   |       796 |         320 |         476 |      0.626 | 0.502 | 0.206 | 0.640 |
|      5 | `resnet50`          |       796 |         320 |         476 |      0.622 | 0.485 | 0.193 | 0.641 |
|      6 | `vgg16`             |       796 |         320 |         476 |      0.618 | 0.495 | 0.191 | 0.609 |
|      7 | `resnet18`          |       796 |         320 |         476 |      0.603 | 0.446 | 0.146 | 0.614 |

## Top5 Minus GT Comparison

Negative deltas mean the detector-driven synthetic training result is lower than the ground-truth synthetic training result.

| Backbone            |   GT MCC |   Top5 MCC |   Delta MCC |   GT AUC |   Top5 AUC |   Delta AUC |   GT F1 |   Top5 F1 |   Delta F1 |   GT Acc |   Top5 Acc |   Delta Acc |
|:--------------------|---------:|-----------:|------------:|---------:|-----------:|------------:|--------:|----------:|-----------:|---------:|-----------:|------------:|
| `efficientnet_v2_s` |    0.607 |      0.409 |      -0.198 |    0.855 |      0.741 |      -0.114 |   0.759 |     0.614 |     -0.145 |    0.813 |      0.722 |      -0.091 |
| `efficientnet_b0`   |    0.568 |      0.343 |      -0.225 |    0.829 |      0.728 |      -0.101 |   0.727 |     0.574 |     -0.153 |    0.795 |      0.692 |      -0.103 |
| `resnet50`          |    0.551 |      0.324 |      -0.227 |    0.853 |      0.702 |      -0.151 |   0.723 |     0.570 |     -0.153 |    0.786 |      0.682 |      -0.104 |
| `densenet121`       |    0.535 |      0.297 |      -0.238 |    0.843 |      0.689 |      -0.154 |   0.712 |     0.565 |     -0.147 |    0.779 |      0.667 |      -0.112 |
| `resnet18`          |    0.504 |      0.252 |      -0.252 |    0.829 |      0.689 |      -0.140 |   0.690 |     0.561 |     -0.129 |    0.765 |      0.637 |      -0.128 |
| `efficientnet_b1`   |    0.540 |      0.263 |      -0.277 |    0.817 |      0.695 |      -0.122 |   0.714 |     0.534 |     -0.180 |    0.781 |      0.653 |      -0.128 |
| `vgg16`             |    0.616 |      0.303 |      -0.313 |    0.865 |      0.688 |      -0.177 |   0.765 |     0.573 |     -0.192 |    0.817 |      0.668 |      -0.149 |

## Half-Backbone-Frozen Top5 Minus GT Comparison

Negative deltas mean the detector-driven synthetic training result is lower than the ground-truth synthetic training result when freezing half the backbone.

| Backbone            |   Half GT MCC |   Half Top5 MCC |   Delta MCC |   Half GT AUC |   Half Top5 AUC |   Delta AUC |   Half GT F1 |   Half Top5 F1 |   Delta F1 |   Half GT Acc |   Half Top5 Acc |   Delta Acc |
|:--------------------|--------------:|----------------:|------------:|--------------:|----------------:|------------:|-------------:|---------------:|-----------:|--------------:|----------------:|------------:|
| `vgg16`             |         0.232 |           0.191 |      -0.042 |         0.663 |           0.609 |      -0.055 |        0.535 |          0.495 |     -0.040 |         0.633 |           0.618 |      -0.015 |
| `resnet18`          |         0.251 |           0.146 |      -0.105 |         0.666 |           0.614 |      -0.052 |        0.559 |          0.446 |     -0.113 |         0.637 |           0.603 |      -0.034 |
| `efficientnet_b0`   |         0.327 |           0.206 |      -0.121 |         0.711 |           0.646 |      -0.065 |        0.584 |          0.503 |     -0.080 |         0.681 |           0.626 |      -0.055 |
| `efficientnet_v2_s` |         0.343 |           0.207 |      -0.136 |         0.711 |           0.623 |      -0.088 |        0.560 |          0.484 |     -0.075 |         0.693 |           0.631 |      -0.063 |
| `efficientnet_b1`   |         0.353 |           0.206 |      -0.147 |         0.739 |           0.640 |      -0.099 |        0.603 |          0.502 |     -0.101 |         0.692 |           0.626 |      -0.067 |
| `densenet121`       |         0.412 |           0.242 |      -0.169 |         0.775 |           0.648 |      -0.126 |        0.635 |          0.518 |     -0.117 |         0.721 |           0.644 |      -0.077 |
| `resnet50`          |         0.379 |           0.193 |      -0.186 |         0.731 |           0.641 |      -0.090 |        0.607 |          0.485 |     -0.122 |         0.707 |           0.622 |      -0.085 |
