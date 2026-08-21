# LUNA16 Synthetic 2D Results: Detector Top4 MinProb 0.5

Questo documento riassume i risultati dei training 2D sulle sintetiche detector-driven generate con `top_k=4` e `min_probability=0.5`, per i metodi RBF e Shepard.

Sorgenti:

- `outputs/luna16_synthetic_2d_top4_minprob0.5_rbf/prediction_metrics.csv`
- `outputs/luna16_synthetic_2d_top4_minprob0.5_shepard/prediction_metrics.csv`

Tutte le metriche principali sotto sono `scope=pooled`: le predizioni dei 10 fold test sono concatenate e valutate insieme. Ogni valutazione pooled contiene 796 campioni: 320 positivi e 476 negativi.

## Executive Summary

Il miglior modello per AUC e' `efficientnet_v2_s` su RBF e `efficientnet_b0` su Shepard. Nel confronto best-model, `RBF` risulta migliore in AUC con delta RBF-Shepard pari a `+0.012`. Sulla media dei backbone, il delta AUC RBF-Shepard e' `+0.010`.

| metodo | backbone migliore | acc | f1 | mcc | auc |
| --- | --- | ---: | ---: | ---: | ---: |
| top4 minprob0.5 RBF | efficientnet_v2_s | 0.685 | 0.581 | 0.332 | 0.712 |
| top4 minprob0.5 Shepard | efficientnet_b0 | 0.678 | 0.582 | 0.322 | 0.700 |

## Pooled Results: RBF

| backbone | acc | f1 | mcc | auc |
| --- | ---: | ---: | ---: | ---: |
| efficientnet_v2_s | 0.685 | 0.581 | 0.332 | 0.712 |
| densenet121 | 0.673 | 0.565 | 0.308 | 0.712 |
| efficientnet_b0 | 0.663 | 0.521 | 0.276 | 0.684 |
| resnet50 | 0.641 | 0.531 | 0.242 | 0.680 |
| vgg16 | 0.653 | 0.535 | 0.263 | 0.674 |
| resnet18 | 0.637 | 0.514 | 0.229 | 0.653 |
| efficientnet_b1 | 0.626 | 0.507 | 0.208 | 0.634 |

Media sui backbone RBF:

| acc | f1 | mcc | auc | best_auc |
| ---: | ---: | ---: | ---: | ---: |
| 0.654 | 0.536 | 0.265 | 0.678 | 0.712 |

## Pooled Results: Shepard

| backbone | acc | f1 | mcc | auc |
| --- | ---: | ---: | ---: | ---: |
| efficientnet_b0 | 0.678 | 0.582 | 0.322 | 0.700 |
| efficientnet_v2_s | 0.685 | 0.556 | 0.325 | 0.684 |
| densenet121 | 0.660 | 0.543 | 0.277 | 0.677 |
| resnet50 | 0.653 | 0.512 | 0.256 | 0.668 |
| efficientnet_b1 | 0.634 | 0.537 | 0.235 | 0.663 |
| resnet18 | 0.652 | 0.525 | 0.258 | 0.662 |
| vgg16 | 0.604 | 0.496 | 0.171 | 0.629 |

Media sui backbone Shepard:

| acc | f1 | mcc | auc | best_auc |
| ---: | ---: | ---: | ---: | ---: |
| 0.652 | 0.536 | 0.263 | 0.669 | 0.700 |

## RBF vs Shepard

Delta = RBF meno Shepard sullo stesso backbone.

| backbone | delta_acc | delta_f1 | delta_mcc | delta_auc |
|---|---:|---:|---:|---:|
| efficientnet_v2_s | +0.000 | +0.025 | +0.007 | +0.028 |
| densenet121 | +0.014 | +0.022 | +0.031 | +0.035 |
| resnet50 | -0.013 | +0.019 | -0.014 | +0.012 |
| vgg16 | +0.049 | +0.039 | +0.093 | +0.045 |
| efficientnet_b0 | -0.015 | -0.060 | -0.046 | -0.017 |
| efficientnet_b1 | -0.009 | -0.031 | -0.028 | -0.029 |
| resnet18 | -0.015 | -0.011 | -0.029 | -0.009 |

RBF ha AUC maggiore di Shepard su 4 backbone, minore su 3 backbone e uguale su 0.

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

Con `top_k=4`, il modello riceve un candidato in piu rispetto a Top3, ma resta ancora abbastanza vicino alla logica di selezione stretta del detector. Questo puo migliorare alcuni casi in cui il quarto candidato recupera informazione utile, ma non elimina la dipendenza dalla qualita del ranking detector.

Il confronto con Top7 e con i mosaici Grad-CAM suggerisce che aumentare il numero di candidati puo recuperare noduli mancati da Top3/Top4/Top5, ma il beneficio dipende da quanto rumore aggiuntivo viene introdotto nella sintetica.

## File Collegati

- Predizioni RBF: `outputs/luna16_synthetic_2d_top4_minprob0.5_rbf/all_test_predictions.csv`
- Predizioni Shepard: `outputs/luna16_synthetic_2d_top4_minprob0.5_shepard/all_test_predictions.csv`
- Mosaici Grad-CAM comparativi: `docs/luna16_synthetic_2d_gradcam_comparison_mosaics_predicted_focused_ct_full/`
- Conclusioni qualitative Grad-CAM: `docs/luna16_synthetic_2d_gradcam_comparison_mosaics_predicted_focused_ct_full/gradcam_mosaic_conclusions.md`
