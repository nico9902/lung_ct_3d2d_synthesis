# LUNA16 Synthetic 2D Latest Output Metrics

This file contains the newly read results from the current `prediction_metrics.csv` files in `outputs/`, kept separate from the historical report in `results/luna16_synthetic_2d.md`.

Sources:
- Ground truth: `outputs/luna16_synthetic_2d_gt/prediction_metrics.csv`
- Detector top5 minprob0.5: `outputs/luna16_synthetic_2d_top5_minprob0.5/prediction_metrics.csv`
- Ground truth, half backbone frozen: `outputs/luna16_synthetic_2d_half_backbone_gt/prediction_metrics.csv`
- Detector top5 minprob0.5, half backbone frozen: `outputs/luna16_synthetic_2d_half_backbone_top5_minprob0.5/prediction_metrics.csv`

Rows below use `scope = pooled`: predictions from all 10 test folds are concatenated for each backbone, then accuracy, F1, MCC, and AUC are recomputed by `src/luna16_synthetic_2d/export_backbone_summary.py` from `all_test_predictions.csv`. Each pooled backbone evaluation contains **796** test samples: **320** positive and **476** negative.

Training was run with `src/luna16_synthetic_2d/run_backbones.sh` and `src/luna16_synthetic_2d/run_backbones_det.sh`: 10 folds, 7 backbones, 100 epochs, 256x384 images, batch size 16, and checkpoint selection monitored by `val_mcc`. The GT runs used `outputs/luna16_saliency_synthetic_gt`; the detector runs used `outputs/luna16_saliency_synthetic_detector_top5_minprob0.5`.

## Latest Pooled Best-Backbone Summary

| Experiment | Best MCC backbone |   Best MCC | Best AUC backbone |   Best AUC |   Samples |   Positives |   Negatives |
|:-----------|:------------------|-----------:|:------------------|-----------:|----------:|------------:|------------:|
| GT synthetic 2D | `efficientnet_v2_s` | 0.618 | `efficientnet_v2_s` | 0.873 | 796 | 320 | 476 |
| Detector top5 minprob0.5 synthetic 2D | `efficientnet_v2_s` | 0.439 | `efficientnet_v2_s` | 0.746 | 796 | 320 | 476 |
| GT synthetic 2D, half backbone frozen | `densenet121` | 0.412 | `densenet121` | 0.775 | 796 | 320 | 476 |
| Detector top5 minprob0.5 synthetic 2D, half backbone frozen | `densenet121` | 0.242 | `densenet121` | 0.648 | 796 | 320 | 476 |

## Latest GT Pooled Metrics

|   Rank | Backbone            |   Accuracy |    F1 |   MCC |   AUC |
|-------:|:--------------------|-----------:|------:|------:|------:|
|      1 | `efficientnet_v2_s` |      0.818 | 0.764 | 0.618 | 0.873 |
|      2 | `vgg16`             |      0.809 | 0.751 | 0.599 | 0.850 |
|      3 | `resnet50`          |      0.793 | 0.727 | 0.563 | 0.860 |
|      4 | `efficientnet_b0`   |      0.784 | 0.711 | 0.544 | 0.828 |
|      5 | `densenet121`       |      0.781 | 0.717 | 0.540 | 0.830 |
|      6 | `resnet18`          |      0.776 | 0.703 | 0.528 | 0.838 |
|      7 | `efficientnet_b1`   |      0.768 | 0.695 | 0.510 | 0.815 |

## Latest Detector Top5 Minprob0.5 Pooled Metrics

|   Rank | Backbone            |   Accuracy |    F1 |   MCC |   AUC |
|-------:|:--------------------|-----------:|------:|------:|------:|
|      1 | `efficientnet_v2_s` |      0.735 | 0.648 | 0.439 | 0.746 |
|      2 | `vgg16`             |      0.692 | 0.588 | 0.347 | 0.710 |
|      3 | `efficientnet_b0`   |      0.692 | 0.574 | 0.343 | 0.728 |
|      4 | `densenet121`       |      0.681 | 0.571 | 0.322 | 0.697 |
|      5 | `resnet18`          |      0.646 | 0.583 | 0.278 | 0.694 |
|      6 | `resnet50`          |      0.656 | 0.545 | 0.271 | 0.681 |
|      7 | `efficientnet_b1`   |      0.653 | 0.548 | 0.268 | 0.686 |

## Latest Top5 Minprob0.5 Minus GT

Negative values mean detector-driven training is worse than GT synthetic training.

| Backbone            |   GT MCC |   Detector MCC |   Delta MCC |   GT AUC |   Detector AUC |   Delta AUC |
|:--------------------|---------:|---------------:|------------:|---------:|---------------:|------------:|
| `efficientnet_v2_s` |    0.618 |          0.439 |      -0.178 |    0.873 |          0.746 |      -0.127 |
| `efficientnet_b0`   |    0.544 |          0.343 |      -0.201 |    0.828 |          0.728 |      -0.100 |
| `densenet121`       |    0.540 |          0.322 |      -0.218 |    0.830 |          0.697 |      -0.133 |
| `efficientnet_b1`   |    0.510 |          0.268 |      -0.242 |    0.815 |          0.686 |      -0.129 |
| `resnet18`          |    0.528 |          0.278 |      -0.251 |    0.838 |          0.694 |      -0.144 |
| `vgg16`             |    0.599 |          0.347 |      -0.251 |    0.850 |          0.710 |      -0.140 |
| `resnet50`          |    0.563 |          0.271 |      -0.292 |    0.860 |          0.681 |      -0.180 |

## Latest Half-Backbone-Frozen Detector Pooled Metrics

|   Rank | Backbone            |   Accuracy |    F1 |   MCC |   AUC |
|-------:|:--------------------|-----------:|------:|------:|------:|
|      1 | `densenet121`       |      0.644 | 0.518 | 0.242 | 0.648 |
|      2 | `efficientnet_v2_s` |      0.631 | 0.484 | 0.207 | 0.623 |
|      3 | `efficientnet_b0`   |      0.626 | 0.503 | 0.206 | 0.646 |
|      4 | `efficientnet_b1`   |      0.626 | 0.502 | 0.206 | 0.640 |
|      5 | `resnet50`          |      0.622 | 0.485 | 0.193 | 0.641 |
|      6 | `vgg16`             |      0.618 | 0.495 | 0.191 | 0.609 |
|      7 | `resnet18`          |      0.603 | 0.446 | 0.146 | 0.614 |
