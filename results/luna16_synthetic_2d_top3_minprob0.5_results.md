# LUNA16 Synthetic 2D Results: Detector Top3 MinProb 0.5

Questo documento riassume i risultati dei training 2D sulle sintetiche detector-driven generate con `top_k=3` e `min_probability=0.5`, per i metodi RBF e Shepard.

Sorgenti:

- `outputs/luna16_synthetic_2d_top3_minprob0.5_rbf/prediction_metrics.csv`
- `outputs/luna16_synthetic_2d_top3_minprob0.5_shepard/prediction_metrics.csv`

Tutte le metriche principali sotto sono `scope=pooled`: le predizioni dei 10 fold test sono concatenate e valutate insieme. Ogni valutazione pooled contiene 796 campioni: 320 positivi e 476 negativi.

## Executive Summary

Il miglior modello per AUC e' `efficientnet_v2_s` su RBF e `efficientnet_b0` su Shepard. Nel confronto best-model, `RBF` risulta migliore in AUC con delta RBF-Shepard pari a `+0.002`. Sulla media dei backbone, il delta AUC RBF-Shepard e' `-0.005`.

| metodo | backbone migliore | acc | f1 | mcc | auc |
| --- | --- | ---: | ---: | ---: | ---: |
| top3 minprob0.5 RBF | efficientnet_v2_s | 0.707 | 0.593 | 0.376 | 0.703 |
| top3 minprob0.5 Shepard | efficientnet_b0 | 0.663 | 0.555 | 0.287 | 0.701 |

## Pooled Results: RBF

| backbone | acc | f1 | mcc | auc |
| --- | ---: | ---: | ---: | ---: |
| efficientnet_v2_s | 0.707 | 0.593 | 0.376 | 0.703 |
| densenet121 | 0.673 | 0.541 | 0.300 | 0.685 |
| resnet18 | 0.667 | 0.519 | 0.283 | 0.676 |
| resnet50 | 0.665 | 0.515 | 0.277 | 0.675 |
| efficientnet_b0 | 0.657 | 0.505 | 0.260 | 0.674 |
| efficientnet_b1 | 0.633 | 0.518 | 0.224 | 0.646 |
| vgg16 | 0.621 | 0.476 | 0.187 | 0.629 |

Media sui backbone RBF:

| acc | f1 | mcc | auc | best_auc |
| ---: | ---: | ---: | ---: | ---: |
| 0.660 | 0.524 | 0.273 | 0.670 | 0.703 |

## Pooled Results: Shepard

| backbone | acc | f1 | mcc | auc |
| --- | ---: | ---: | ---: | ---: |
| efficientnet_b0 | 0.663 | 0.555 | 0.287 | 0.701 |
| efficientnet_v2_s | 0.690 | 0.583 | 0.341 | 0.693 |
| resnet50 | 0.658 | 0.516 | 0.266 | 0.681 |
| efficientnet_b1 | 0.642 | 0.547 | 0.251 | 0.679 |
| densenet121 | 0.649 | 0.536 | 0.258 | 0.665 |
| vgg16 | 0.618 | 0.532 | 0.210 | 0.654 |
| resnet18 | 0.613 | 0.519 | 0.195 | 0.648 |

Media sui backbone Shepard:

| acc | f1 | mcc | auc | best_auc |
| ---: | ---: | ---: | ---: | ---: |
| 0.648 | 0.541 | 0.258 | 0.675 | 0.701 |

## RBF vs Shepard

Delta = RBF meno Shepard sullo stesso backbone.

| backbone | delta_acc | delta_f1 | delta_mcc | delta_auc |
|---|---:|---:|---:|---:|
| efficientnet_v2_s | +0.018 | +0.010 | +0.034 | +0.009 |
| densenet121 | +0.024 | +0.005 | +0.042 | +0.020 |
| resnet50 | +0.006 | -0.001 | +0.012 | -0.006 |
| vgg16 | +0.003 | -0.057 | -0.023 | -0.024 |
| efficientnet_b0 | -0.006 | -0.050 | -0.027 | -0.027 |
| efficientnet_b1 | -0.009 | -0.029 | -0.027 | -0.033 |
| resnet18 | +0.054 | +0.000 | +0.088 | +0.027 |

RBF ha AUC maggiore di Shepard su 3 backbone, minore su 4 backbone e uguale su 0.

## Context: Top3, Top4, Top5 And Top7

| esperimento | mean_acc | mean_f1 | mean_mcc | mean_auc | best_auc | best backbone |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Top3 RBF | 0.660 | 0.524 | 0.273 | 0.670 | 0.703 | efficientnet_v2_s |
| Top3 Shepard | 0.648 | 0.541 | 0.258 | 0.675 | 0.701 | efficientnet_b0 |
| Top4 RBF | 0.654 | 0.536 | 0.265 | 0.678 | 0.712 | efficientnet_v2_s |
| Top4 Shepard | 0.652 | 0.536 | 0.263 | 0.669 | 0.700 | efficientnet_b0 |
| Top5 Shepard | 0.668 | 0.560 | 0.298 | 0.688 | 0.729 | efficientnet_v2_s |
| Top7 RBF | 0.666 | 0.578 | 0.303 | 0.706 | 0.734 | efficientnet_v2_s |
| Top7 Shepard | 0.658 | 0.565 | 0.285 | 0.689 | 0.730 | efficientnet_v2_s |

## Lettura Complessiva

Con `top_k=3`, il numero di candidati e' piu contenuto: questo riduce il rumore introdotto dal detector, ma aumenta il rischio di non includere il nodulo reale nei casi in cui il detector non lo posiziona tra i primissimi candidati.

I risultati vanno letti insieme ai mosaici Grad-CAM: quando il candidato corretto manca, la sintetica puo diventare visivamente saliente ma poco informativa per la classificazione benigno/maligno. Questo limita soprattutto i casi maligni, dove perdere il nodulo significa perdere il segnale discriminativo principale.

## File Collegati

- Predizioni RBF: `outputs/luna16_synthetic_2d_top3_minprob0.5_rbf/all_test_predictions.csv`
- Predizioni Shepard: `outputs/luna16_synthetic_2d_top3_minprob0.5_shepard/all_test_predictions.csv`
- Mosaici Grad-CAM comparativi: `docs/luna16_synthetic_2d_gradcam_comparison_mosaics_predicted_focused_ct_full/`
- Conclusioni qualitative Grad-CAM: `docs/luna16_synthetic_2d_gradcam_comparison_mosaics_predicted_focused_ct_full/gradcam_mosaic_conclusions.md`
