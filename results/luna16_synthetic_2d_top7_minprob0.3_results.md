# LUNA16 Synthetic 2D Results: Detector Top7 MinProb 0.3

Questo documento riassume i nuovi risultati dei training 2D sulle sintetiche detector-driven generate con `top_k=7` e `min_probability=0.3`, per i metodi RBF e Shepard.

Sorgenti:

- `outputs/luna16_synthetic_2d_top7_minprob0.3_rbf/prediction_metrics.csv`
- `outputs/luna16_synthetic_2d_top7_minprob0.3_shepard/prediction_metrics.csv`

Tutte le metriche principali sotto sono `scope=pooled`: le predizioni dei 10 fold test sono concatenate e valutate insieme. Ogni valutazione pooled contiene 796 campioni: 320 positivi e 476 negativi.

## Executive Summary

Il miglior modello sui nuovi esperimenti e' `efficientnet_v2_s` in entrambi i metodi:

| metodo | backbone migliore | acc | f1 | mcc | auc |
|---|---|---:|---:|---:|---:|
| top7 minprob0.3 RBF | efficientnet_v2_s | 0.700 | 0.635 | 0.381 | 0.734 |
| top7 minprob0.3 Shepard | efficientnet_v2_s | 0.673 | 0.585 | 0.316 | 0.730 |

RBF e' complessivamente migliore di Shepard sui risultati classificativi top7/minprob0.3, nonostante il QC geometrico avesse indicato superfici RBF piu' rumorose. La differenza e' piccola sul best AUC (`+0.004`), ma piu' visibile su accuracy, F1 e MCC per `efficientnet_v2_s`.

## Pooled Results: RBF

| backbone | acc | f1 | mcc | auc |
|---|---:|---:|---:|---:|
| efficientnet_v2_s | 0.700 | 0.635 | 0.381 | 0.734 |
| densenet121 | 0.675 | 0.587 | 0.319 | 0.719 |
| resnet50 | 0.682 | 0.576 | 0.326 | 0.709 |
| vgg16 | 0.668 | 0.574 | 0.303 | 0.703 |
| efficientnet_b0 | 0.655 | 0.570 | 0.281 | 0.702 |
| efficientnet_b1 | 0.641 | 0.565 | 0.260 | 0.692 |
| resnet18 | 0.642 | 0.540 | 0.248 | 0.681 |

Media sui backbone RBF:

| acc | f1 | mcc | auc | best_auc |
|---:|---:|---:|---:|---:|
| 0.666 | 0.578 | 0.303 | 0.706 | 0.734 |

## Pooled Results: Shepard

| backbone | acc | f1 | mcc | auc |
|---|---:|---:|---:|---:|
| efficientnet_v2_s | 0.673 | 0.585 | 0.316 | 0.730 |
| densenet121 | 0.661 | 0.581 | 0.296 | 0.698 |
| resnet50 | 0.667 | 0.576 | 0.302 | 0.698 |
| efficientnet_b1 | 0.662 | 0.586 | 0.300 | 0.687 |
| efficientnet_b0 | 0.657 | 0.556 | 0.278 | 0.683 |
| vgg16 | 0.644 | 0.552 | 0.257 | 0.668 |
| resnet18 | 0.644 | 0.518 | 0.242 | 0.660 |

Media sui backbone Shepard:

| acc | f1 | mcc | auc | best_auc |
|---:|---:|---:|---:|---:|
| 0.659 | 0.565 | 0.285 | 0.689 | 0.730 |

## RBF vs Shepard

Delta = RBF meno Shepard sullo stesso backbone.

| backbone | delta_acc | delta_f1 | delta_mcc | delta_auc |
|---|---:|---:|---:|---:|
| densenet121 | +0.014 | +0.006 | +0.023 | +0.021 |
| efficientnet_b0 | -0.003 | +0.014 | +0.003 | +0.019 |
| efficientnet_b1 | -0.021 | -0.020 | -0.041 | +0.005 |
| efficientnet_v2_s | +0.026 | +0.050 | +0.065 | +0.004 |
| resnet18 | -0.003 | +0.022 | +0.005 | +0.022 |
| resnet50 | +0.015 | +0.000 | +0.024 | +0.011 |
| vgg16 | +0.024 | +0.023 | +0.046 | +0.035 |

RBF ha AUC maggiore di Shepard su tutti i backbone. Shepard ha un vantaggio minimo in accuracy solo su `efficientnet_b0`, `efficientnet_b1` e `resnet18`, ma non in AUC.

## Context: Top5 MinProb 0.5 And GT

| esperimento | mean_acc | mean_f1 | mean_mcc | mean_auc | best_auc | best backbone |
|---|---:|---:|---:|---:|---:|---|
| GT no contour | 0.762 | 0.686 | 0.498 | 0.816 | 0.839 | efficientnet_v2_s |
| GT Shepard | 0.790 | 0.724 | 0.558 | 0.843 | 0.856 | resnet50 |
| detector top5 minprob0.5 RBF/no contour | 0.654 | 0.550 | 0.272 | 0.683 | 0.730 | efficientnet_v2_s |
| detector top5 minprob0.5 Shepard | 0.668 | 0.560 | 0.298 | 0.688 | 0.729 | efficientnet_v2_s |
| detector top7 minprob0.3 RBF | 0.666 | 0.578 | 0.303 | 0.706 | 0.734 | efficientnet_v2_s |
| detector top7 minprob0.3 Shepard | 0.659 | 0.565 | 0.285 | 0.689 | 0.730 | efficientnet_v2_s |

Rispetto ai detector top5/minprob0.5:

- Top7/minprob0.3 RBF migliora il mean AUC da `0.683` a `0.706` e il best AUC da `0.730` a `0.734`.
- Top7/minprob0.3 Shepard resta sostanzialmente allineato al top5 Shepard: mean AUC `0.689` vs `0.688`, best AUC `0.730` vs `0.729`.
- Entrambi i detector-driven restano nettamente sotto le sintetiche GT, specialmente rispetto al GT Shepard.

## Lettura Complessiva

I nuovi risultati suggeriscono che abbassare la soglia detector a `0.3` e usare `top7` puo' essere utile per RBF in termini classificativi, pur generando superfici piu' complesse nel QC geometrico.

Questo e' coerente con una possibile tensione tra qualita' geometrico-visiva e informazione discriminativa: le superfici top7 RBF sono piu' rumorose, ma potrebbero includere piu' segnali detector utili al classificatore.

Per Shepard, il passaggio top5/minprob0.5 -> top7/minprob0.3 non porta un guadagno chiaro: il best model resta praticamente uguale in AUC, e la media sui backbone cambia poco.

## File Collegati

- QC top7/minprob0.3: `docs/luna16_saliency_synthetic_detector_top7_minprob03_qc_comparison.md`
- Mosaici GT vs RBF/Shepard: `docs/luna16_gt_vs_top7_synthetic_overlay_mosaics/`
- Campioni organizzati per nodulo: `docs/luna16_gt_vs_top7_synthetic_by_sample/`
- Predizioni RBF: `outputs/luna16_synthetic_2d_top7_minprob0.3_rbf/all_test_predictions.csv`
- Predizioni Shepard: `outputs/luna16_synthetic_2d_top7_minprob0.3_shepard/all_test_predictions.csv`
