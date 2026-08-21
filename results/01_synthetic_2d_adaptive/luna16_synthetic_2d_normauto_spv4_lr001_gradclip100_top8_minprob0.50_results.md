# Risultati backbone 2D - nuovo detector, top8, threshold 0.50

Data controllo server: 2026-08-02

Esperimenti:

- `luna16_synthetic_2d_normauto_spv4_lr001_gradclip100_top8_minprob0.50_rbf`
- `luna16_synthetic_2d_normauto_spv4_lr001_gradclip100_top8_minprob0.50_shepard`

Entrambi gli esperimenti sono terminati e hanno prodotto 70 file `test_predictions.csv`.
L'export finale in Excel e' fallito per assenza di `openpyxl`, ma i file `prediction_metrics.csv` sono stati generati correttamente.

## Risultati pooled - RBF

| backbone | acc | f1 | mcc | auc |
|---|---:|---:|---:|---:|
| densenet121 | 0.715 | 0.634 | 0.402 | 0.771 |
| efficientnet_b0 | 0.712 | 0.614 | 0.390 | 0.764 |
| efficientnet_b1 | 0.711 | 0.618 | 0.389 | 0.754 |
| efficientnet_v2_s | 0.730 | 0.645 | 0.430 | 0.782 |
| resnet18 | 0.719 | 0.630 | 0.406 | 0.756 |
| resnet50 | 0.715 | 0.590 | 0.390 | 0.749 |
| vgg16 | 0.725 | 0.651 | 0.424 | 0.744 |

Media sui backbone:

| acc | f1 | mcc | auc |
|---:|---:|---:|---:|
| 0.718 | 0.626 | 0.404 | 0.760 |

Migliore configurazione RBF:

- best MCC: `efficientnet_v2_s`, MCC 0.430
- best AUC: `efficientnet_v2_s`, AUC 0.782

## Risultati pooled - Shepard

| backbone | acc | f1 | mcc | auc |
|---|---:|---:|---:|---:|
| densenet121 | 0.731 | 0.649 | 0.433 | 0.788 |
| efficientnet_b0 | 0.707 | 0.616 | 0.382 | 0.756 |
| efficientnet_b1 | 0.683 | 0.590 | 0.333 | 0.736 |
| efficientnet_v2_s | 0.731 | 0.630 | 0.429 | 0.769 |
| resnet18 | 0.709 | 0.605 | 0.381 | 0.746 |
| resnet50 | 0.736 | 0.643 | 0.441 | 0.788 |
| vgg16 | 0.701 | 0.625 | 0.376 | 0.752 |

Media sui backbone:

| acc | f1 | mcc | auc |
|---:|---:|---:|---:|
| 0.714 | 0.623 | 0.396 | 0.762 |

Migliore configurazione Shepard:

- best MCC: `resnet50`, MCC 0.441
- best AUC: `densenet121`, AUC 0.788

## Confronto con i migliori precedenti

| setup | best backbone | best MCC | best AUC |
|---|---|---:|---:|
| precedente `fold8retry 0.45/top8 shepard` | `efficientnet_v2_s` | 0.457 | 0.782 |
| precedente `fold8retry 0.45/top8 rbf` | `efficientnet_v2_s` | 0.442 | 0.788 |
| nuovo detector `0.50/top8 shepard` | `resnet50` | 0.441 | 0.788 |
| nuovo detector `0.50/top8 rbf` | `efficientnet_v2_s` | 0.430 | 0.782 |

## Interpretazione

Il punto di lavoro `threshold=0.50, topk=8` migliora chiaramente rispetto al tentativo con `threshold=0.45, topk=8`, ma non supera i migliori risultati precedenti.

Il miglior risultato nuovo e' `shepard + resnet50`, con MCC 0.441 e AUC 0.788. Rimane sotto al best precedente `fold8retry 0.45/top8 shepard + efficientnet_v2_s`, che aveva MCC 0.457.

Conclusione: il nuovo detector migliora la FROC, ma questo non si traduce automaticamente in un miglioramento downstream sulle backbone 2D. Probabilmente il detector copre piu' noduli o li copre meglio in termini FROC, ma il set di candidati sintetici generato non e' piu' informativo del precedente per la classificazione 2D.

Best operativo attuale:

- `fold8retry 0.45/top8 shepard`
- backbone `efficientnet_v2_s`
- MCC 0.457
- AUC 0.782

