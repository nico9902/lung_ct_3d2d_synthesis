# LUNA16 Synthetic 2D Results: NormAuto SPV4 LR0.01 GradClip100 Top8 MinProb 0.45

Questo documento riassume i training 2D sulle sintetiche generate con il detector:

`outputs/scpmnet_luna16_10fold_normauto_amp_spv4_lr001_randomcrop_gradclip100_merged`

Punto di lavoro detector: `threshold=0.45`, `top_k=8`, copertura noduli `85.67%`.

Sorgenti:

- RBF metrics: `outputs/luna16_synthetic_2d_normauto_spv4_lr001_gradclip100_top8_minprob0.45_rbf/prediction_metrics.csv`
- Shepard metrics: `outputs/luna16_synthetic_2d_normauto_spv4_lr001_gradclip100_top8_minprob0.45_shepard/prediction_metrics.csv`
- RBF images server: `/ssd2/domenico/datasets/synthetic_2d/luna16_saliency_synthetic_detector_normauto_spv4_lr001_gradclip100_top8_minprob0.45_rbf`
- Shepard images server: `/ssd2/domenico/datasets/synthetic_2d/luna16_saliency_synthetic_detector_normauto_spv4_lr001_gradclip100_top8_minprob0.45_shepard`

Nota operativa: i 70 training per esperimento sono completati. L'export finale `.xlsx` e' fallito per `openpyxl` mancante sul server, ma i CSV aggregati sono stati creati correttamente.

## Executive Summary

Il miglior risultato RBF e' con `efficientnet_v2_s`: MCC `0.395`, AUC `0.761`.

Il miglior risultato Shepard per MCC e' con `efficientnet_b0`: MCC `0.393`, AUC `0.741`. Il miglior AUC Shepard e' invece con `resnet50`: AUC `0.760`, MCC `0.373`.

Sulla media dei backbone, Shepard e' leggermente migliore di RBF:

| metodo | mean acc | mean f1 | mean mcc | mean auc |
|---|---:|---:|---:|---:|
| RBF | 0.689 | 0.592 | 0.343 | 0.732 |
| Shepard | 0.693 | 0.599 | 0.353 | 0.740 |

## Pooled Results: RBF

| backbone | acc | f1 | mcc | auc |
|---|---:|---:|---:|---:|
| efficientnet_v2_s | 0.715 | 0.615 | 0.395 | 0.761 |
| efficientnet_b1 | 0.693 | 0.614 | 0.360 | 0.728 |
| resnet50 | 0.690 | 0.589 | 0.343 | 0.725 |
| efficientnet_b0 | 0.687 | 0.594 | 0.341 | 0.732 |
| densenet121 | 0.680 | 0.595 | 0.330 | 0.740 |
| resnet18 | 0.683 | 0.562 | 0.324 | 0.729 |
| vgg16 | 0.672 | 0.574 | 0.309 | 0.711 |

## Pooled Results: Shepard

| backbone | acc | f1 | mcc | auc |
|---|---:|---:|---:|---:|
| efficientnet_b0 | 0.712 | 0.623 | 0.393 | 0.741 |
| resnet50 | 0.705 | 0.601 | 0.373 | 0.760 |
| efficientnet_v2_s | 0.704 | 0.604 | 0.372 | 0.733 |
| densenet121 | 0.691 | 0.593 | 0.347 | 0.744 |
| efficientnet_b1 | 0.685 | 0.602 | 0.341 | 0.741 |
| resnet18 | 0.686 | 0.579 | 0.334 | 0.739 |
| vgg16 | 0.668 | 0.594 | 0.314 | 0.723 |

## Fold Variability

Media e deviazione standard sui 10 fold, calcolate dalle metriche per-fold.

| metodo | backbone | acc mean | f1 mean | mcc mean | mcc std | auc mean | auc std |
|---|---|---:|---:|---:|---:|---:|---:|
| RBF | efficientnet_v2_s | 0.715 | 0.613 | 0.400 | 0.090 | 0.760 | 0.060 |
| RBF | efficientnet_b1 | 0.693 | 0.609 | 0.366 | 0.068 | 0.735 | 0.048 |
| RBF | efficientnet_b0 | 0.687 | 0.588 | 0.347 | 0.094 | 0.745 | 0.051 |
| RBF | resnet50 | 0.689 | 0.586 | 0.341 | 0.101 | 0.727 | 0.066 |
| RBF | densenet121 | 0.680 | 0.594 | 0.333 | 0.108 | 0.743 | 0.056 |
| RBF | resnet18 | 0.683 | 0.553 | 0.316 | 0.091 | 0.730 | 0.046 |
| RBF | vgg16 | 0.672 | 0.569 | 0.309 | 0.104 | 0.730 | 0.057 |
| Shepard | efficientnet_b0 | 0.713 | 0.619 | 0.399 | 0.066 | 0.756 | 0.038 |
| Shepard | efficientnet_v2_s | 0.704 | 0.598 | 0.381 | 0.073 | 0.745 | 0.027 |
| Shepard | resnet50 | 0.705 | 0.600 | 0.376 | 0.088 | 0.761 | 0.040 |
| Shepard | densenet121 | 0.691 | 0.592 | 0.355 | 0.081 | 0.749 | 0.052 |
| Shepard | efficientnet_b1 | 0.685 | 0.599 | 0.346 | 0.078 | 0.752 | 0.030 |
| Shepard | resnet18 | 0.686 | 0.578 | 0.344 | 0.122 | 0.752 | 0.065 |
| Shepard | vgg16 | 0.668 | 0.592 | 0.329 | 0.083 | 0.749 | 0.079 |

## Comparison With Previous Top8 MinProb 0.45

Confronto rispetto alle sintetiche precedenti generate con `outputs/scpmnet_luna16_10fold_normauto_with_fold8_retry`.

| esperimento | best backbone | best mcc | best auc | mean mcc | mean auc |
|---|---|---:|---:|---:|---:|
| Previous RBF fold8retry | efficientnet_v2_s | 0.442 | 0.788 | 0.371 | 0.738 |
| New RBF spv4 gradclip100 | efficientnet_v2_s | 0.395 | 0.761 | 0.343 | 0.732 |
| Previous Shepard fold8retry | efficientnet_v2_s | 0.457 | 0.782 | 0.347 | 0.729 |
| New Shepard spv4 gradclip100 | efficientnet_b0 | 0.393 | 0.741 | 0.353 | 0.740 |

Lettura:

- RBF peggiora rispetto al precedente fold8retry: best MCC `0.442 -> 0.395`, best AUC `0.788 -> 0.761`.
- Shepard perde sul miglior singolo backbone: best MCC `0.457 -> 0.393`.
- Shepard pero' migliora leggermente come media sui backbone: mean MCC `0.347 -> 0.353`, mean AUC `0.729 -> 0.740`.
- Il nuovo detector ha FROC migliore, ma questo non si traduce automaticamente in migliori performance 2D: il punto `top8/minprob0.45` con il detector spv4 gradclip100 sembra produrre sintetiche piu' stabili in media per Shepard, ma meno forti del precedente fold8retry sul miglior backbone.

