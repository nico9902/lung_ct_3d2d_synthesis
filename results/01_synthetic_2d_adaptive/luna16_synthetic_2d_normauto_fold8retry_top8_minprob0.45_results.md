# LUNA16 Synthetic 2D Results: NormAuto Fold8 Retry Top8 MinProb 0.45

Questo documento riassume i risultati dei nuovi training 2D sulle sintetiche detector-driven generate con il detector `normauto_with_fold8_retry`, usando `top_k=8` e `min_probability=0.45`.

Sorgenti sul server:

- Shepard metrics: `outputs/luna16_synthetic_2d_normauto_fold8retry_top8_minprob0.45_shepard/prediction_metrics.csv`
- RBF metrics: `outputs/luna16_synthetic_2d_normauto_fold8retry_top8_minprob0.45_rbf/prediction_metrics.csv`
- Shepard images: `/ssd2/domenico/datasets/synthetic_2d/luna16_saliency_synthetic_detector_normauto_fold8retry_top8_minprob0.45_shepard`
- RBF images: `/ssd2/domenico/datasets/synthetic_2d/luna16_saliency_synthetic_detector_normauto_fold8retry_top8_minprob0.45_rbf`

Tutte le metriche principali sotto sono `scope=pooled`: le predizioni dei 10 fold test sono concatenate e valutate insieme. Ogni valutazione pooled contiene **796** campioni: **320 positivi** e **476 negativi**.

## Executive Summary

Il miglior backbone e' `efficientnet_v2_s` per entrambe le interpolazioni.

| metodo | backbone migliore | acc | f1 | mcc | auc |
|---|---|---:|---:|---:|---:|
| RBF | efficientnet_v2_s | 0.735 | 0.655 | 0.442 | 0.788 |
| Shepard | efficientnet_v2_s | 0.742 | 0.661 | 0.457 | 0.782 |

La differenza e' molto vicina sul best model: Shepard ha accuracy, F1 e MCC leggermente migliori con `efficientnet_v2_s`, mentre RBF ha AUC leggermente migliore (`+0.007`). Sulla media dei backbone, RBF e' complessivamente avanti.

| metodo | mean acc | mean f1 | mean mcc | mean auc |
|---|---:|---:|---:|---:|
| RBF | 0.703 | 0.604 | 0.371 | 0.738 |
| Shepard | 0.690 | 0.593 | 0.347 | 0.729 |

## Pooled Results: RBF

| backbone | acc | f1 | mcc | auc |
|---|---:|---:|---:|---:|
| efficientnet_v2_s | 0.735 | 0.655 | 0.442 | 0.788 |
| densenet121 | 0.697 | 0.595 | 0.358 | 0.748 |
| resnet18 | 0.695 | 0.594 | 0.354 | 0.737 |
| resnet50 | 0.696 | 0.603 | 0.359 | 0.727 |
| vgg16 | 0.715 | 0.626 | 0.398 | 0.727 |
| efficientnet_b0 | 0.681 | 0.575 | 0.324 | 0.720 |
| efficientnet_b1 | 0.702 | 0.576 | 0.363 | 0.716 |

## Pooled Results: Shepard

| backbone | acc | f1 | mcc | auc |
|---|---:|---:|---:|---:|
| efficientnet_v2_s | 0.742 | 0.661 | 0.457 | 0.782 |
| densenet121 | 0.692 | 0.591 | 0.348 | 0.731 |
| efficientnet_b1 | 0.697 | 0.608 | 0.363 | 0.728 |
| vgg16 | 0.666 | 0.610 | 0.321 | 0.725 |
| efficientnet_b0 | 0.688 | 0.574 | 0.336 | 0.724 |
| resnet18 | 0.683 | 0.562 | 0.324 | 0.721 |
| resnet50 | 0.663 | 0.541 | 0.282 | 0.696 |

## RBF vs Shepard

Delta = RBF meno Shepard sullo stesso backbone.

| backbone | delta acc | delta f1 | delta mcc | delta auc |
|---|---:|---:|---:|---:|
| resnet50 | +0.033 | +0.062 | +0.077 | +0.031 |
| densenet121 | +0.005 | +0.004 | +0.010 | +0.017 |
| resnet18 | +0.011 | +0.032 | +0.029 | +0.016 |
| efficientnet_v2_s | -0.008 | -0.006 | -0.015 | +0.007 |
| vgg16 | +0.049 | +0.016 | +0.077 | +0.003 |
| efficientnet_b0 | -0.008 | +0.001 | -0.013 | -0.004 |
| efficientnet_b1 | +0.005 | -0.032 | -0.000 | -0.012 |

RBF migliora l'AUC su 5 backbone su 7. Shepard resta migliore su `efficientnet_b0` e `efficientnet_b1` per AUC, e sul miglior backbone (`efficientnet_v2_s`) e' leggermente migliore in MCC.

## Detector Baseline Comparison

Le sintetiche di questo esperimento sono state generate con il detector `outputs/scpmnet_luna16_10fold_normauto_with_fold8_retry`, cioe' il detector 10-fold normauto in cui la fold8 e' stata rifatta. Il confronto sotto e' rispetto alla baseline detector precedente `outputs/scpmnet_luna16_10fold`.

| FP/scan | baseline sensitivity | normauto fold8retry sensitivity | delta |
|---:|---:|---:|---:|
| 0.125 | 0.000 | 0.365 | +0.365 |
| 0.25 | 0.017 | 0.476 | +0.459 |
| 0.5 | 0.346 | 0.589 | +0.243 |
| 1.0 | 0.576 | 0.701 | +0.125 |
| 2.0 | 0.705 | 0.781 | +0.076 |
| 4.0 | 0.775 | 0.834 | +0.059 |
| 8.0 | 0.825 | 0.874 | +0.049 |

Il mean FROC pooled passa da **0.463** a **0.660**, con un miglioramento assoluto di **+0.196**. Il guadagno principale e' ai bassi FP/scan: a `0.125` e `0.25` FP/scan la baseline era quasi cieca, mentre il detector normauto fold8retry recupera rispettivamente `0.365` e `0.476` di sensitivity.

## Fold Variability

Media e deviazione standard sui 10 fold, calcolate dalle metriche per-fold.

| metodo | backbone | auc mean | auc std | mcc mean | mcc std |
|---|---|---:|---:|---:|---:|
| RBF | efficientnet_v2_s | 0.788 | 0.044 | 0.439 | 0.089 |
| RBF | densenet121 | 0.759 | 0.036 | 0.355 | 0.119 |
| RBF | resnet18 | 0.740 | 0.039 | 0.354 | 0.089 |
| RBF | resnet50 | 0.732 | 0.053 | 0.362 | 0.119 |
| RBF | vgg16 | 0.746 | 0.062 | 0.409 | 0.098 |
| RBF | efficientnet_b0 | 0.728 | 0.045 | 0.327 | 0.122 |
| RBF | efficientnet_b1 | 0.725 | 0.048 | 0.371 | 0.084 |
| Shepard | efficientnet_v2_s | 0.780 | 0.063 | 0.453 | 0.088 |
| Shepard | densenet121 | 0.739 | 0.046 | 0.350 | 0.112 |
| Shepard | efficientnet_b1 | 0.728 | 0.051 | 0.348 | 0.126 |
| Shepard | vgg16 | 0.733 | 0.060 | 0.336 | 0.106 |
| Shepard | efficientnet_b0 | 0.730 | 0.046 | 0.327 | 0.089 |
| Shepard | resnet18 | 0.733 | 0.048 | 0.325 | 0.113 |
| Shepard | resnet50 | 0.699 | 0.052 | 0.290 | 0.109 |

## Comparison With Previous Detector-Driven Runs

Rispetto agli esperimenti precedenti con detector-driven surfaces, il nuovo punto di lavoro `top8/minprob0.45` e il detector `normauto_with_fold8_retry` migliorano in modo netto il miglior AUC.

| esperimento | best backbone | best mcc | best auc |
|---|---|---:|---:|
| detector top7/minprob0.3 RBF | efficientnet_v2_s | 0.381 | 0.734 |
| detector top7/minprob0.3 Shepard | efficientnet_v2_s | 0.316 | 0.730 |
| detector normauto fold8retry top8/minprob0.45 RBF | efficientnet_v2_s | 0.442 | 0.788 |
| detector normauto fold8retry top8/minprob0.45 Shepard | efficientnet_v2_s | 0.457 | 0.782 |

Quindi, rispetto al vecchio top7/minprob0.3:

- RBF: best AUC `0.734 -> 0.788` (`+0.054`), best MCC `0.381 -> 0.442` (`+0.061`).
- Shepard: best AUC `0.730 -> 0.782` (`+0.052`), best MCC `0.316 -> 0.457` (`+0.141`).

## Lettura Complessiva

Il nuovo detector e il punto di lavoro `top8/minprob0.45` producono sintetiche molto piu' utili per il task 2D rispetto ai detector-driven esperimenti precedenti. La scelta RBF vs Shepard non e' univoca: RBF e' piu' forte come media sui backbone e come best AUC, mentre Shepard produce il miglior MCC assoluto con `efficientnet_v2_s`.

Come punto operativo, se l'obiettivo e' massimizzare AUC sceglierei **RBF + efficientnet_v2_s**. Se invece vuoi privilegiare MCC/F1 del miglior singolo backbone, sceglierei **Shepard + efficientnet_v2_s**.

## File Collegati

- RBF predictions: `outputs/luna16_synthetic_2d_normauto_fold8retry_top8_minprob0.45_rbf/all_test_predictions.csv`
- Shepard predictions: `outputs/luna16_synthetic_2d_normauto_fold8retry_top8_minprob0.45_shepard/all_test_predictions.csv`
- RBF summary: `outputs/luna16_synthetic_2d_normauto_fold8retry_top8_minprob0.45_rbf/prediction_metrics.csv`
- Shepard summary: `outputs/luna16_synthetic_2d_normauto_fold8retry_top8_minprob0.45_shepard/prediction_metrics.csv`
