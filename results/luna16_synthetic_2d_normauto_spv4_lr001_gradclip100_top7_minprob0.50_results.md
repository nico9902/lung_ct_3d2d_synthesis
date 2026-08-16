# Risultati backbone 2D - nuovo detector, top7, threshold 0.50

Data controllo server: 2026-08-02

Esperimenti:

- `luna16_synthetic_2d_normauto_spv4_lr001_gradclip100_top7_minprob0.50_rbf`
- `luna16_synthetic_2d_normauto_spv4_lr001_gradclip100_top7_minprob0.50_shepard`

Entrambi gli esperimenti sono terminati e hanno prodotto 70 file `test_predictions.csv`.
L'export finale in Excel e' fallito per assenza di `openpyxl`, ma i file `prediction_metrics.csv` sono stati generati correttamente.

## Risultati pooled - RBF

| backbone | acc | f1 | mcc | auc |
|---|---:|---:|---:|---:|
| densenet121 | 0.706 | 0.633 | 0.388 | 0.768 |
| efficientnet_b0 | 0.705 | 0.608 | 0.375 | 0.746 |
| efficientnet_b1 | 0.697 | 0.629 | 0.373 | 0.740 |
| efficientnet_v2_s | 0.758 | 0.668 | 0.486 | 0.794 |
| resnet18 | 0.682 | 0.565 | 0.322 | 0.731 |
| resnet50 | 0.720 | 0.624 | 0.406 | 0.771 |
| vgg16 | 0.735 | 0.667 | 0.447 | 0.762 |

Media sui backbone:

| acc | f1 | mcc | auc |
|---:|---:|---:|---:|
| 0.715 | 0.628 | 0.400 | 0.759 |

Migliore configurazione RBF:

- best MCC: `efficientnet_v2_s`, MCC 0.486
- best AUC: `efficientnet_v2_s`, AUC 0.794

## Risultati pooled - Shepard

| backbone | acc | f1 | mcc | auc |
|---|---:|---:|---:|---:|
| densenet121 | 0.714 | 0.635 | 0.400 | 0.778 |
| efficientnet_b0 | 0.716 | 0.623 | 0.399 | 0.760 |
| efficientnet_b1 | 0.673 | 0.582 | 0.315 | 0.723 |
| efficientnet_v2_s | 0.746 | 0.649 | 0.461 | 0.782 |
| resnet18 | 0.686 | 0.595 | 0.340 | 0.745 |
| resnet50 | 0.732 | 0.647 | 0.435 | 0.786 |
| vgg16 | 0.724 | 0.649 | 0.421 | 0.764 |

Media sui backbone:

| acc | f1 | mcc | auc |
|---:|---:|---:|---:|
| 0.713 | 0.626 | 0.396 | 0.763 |

Migliore configurazione Shepard:

- best MCC: `efficientnet_v2_s`, MCC 0.461
- best AUC: `resnet50`, AUC 0.786

## Confronto con i migliori precedenti

| setup | best backbone | best MCC | best AUC |
|---|---|---:|---:|
| precedente `fold8retry 0.45/top8 shepard` | `efficientnet_v2_s` | 0.457 | 0.782 |
| precedente `fold8retry 0.45/top8 rbf` | `efficientnet_v2_s` | 0.442 | 0.788 |
| nuovo detector `0.50/top8 shepard` | `resnet50` | 0.441 | 0.788 |
| nuovo detector `0.50/top8 rbf` | `efficientnet_v2_s` | 0.430 | 0.782 |
| nuovo detector `0.50/top7 shepard` | `efficientnet_v2_s` | 0.461 | 0.782 |
| nuovo detector `0.50/top7 rbf` | `efficientnet_v2_s` | 0.486 | 0.794 |

## Interpretazione

Il punto di lavoro `threshold=0.50, topk=7` e' il migliore finora in termini di MCC.
La configurazione `rbf + efficientnet_v2_s` raggiunge MCC 0.486, superando il precedente best MCC 0.457.

L'AUC pero' resta appena sotto il target 0.80: il miglior valore osservato e' 0.794 con `rbf + efficientnet_v2_s`.

Rispetto a `threshold=0.50, topk=8`, ridurre il top-k da 8 a 7 sembra aver eliminato candidati marginali/noisy e ha migliorato molto la qualita' decisionale del classificatore, soprattutto su MCC e F1.

Best operativo aggiornato:

- sintetiche `threshold=0.50, topk=7`
- interpolazione `rbf`
- backbone `efficientnet_v2_s`
- MCC 0.486
- AUC 0.794

