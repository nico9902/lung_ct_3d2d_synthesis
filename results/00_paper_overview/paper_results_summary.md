# Paper Results Summary

## Central comparison

Il confronto principale da portare nel paper e':

| Block | Representation | Model | AUC | MCC | F1 | Status |
|---|---|---|---:|---:|---:|---|
| Proposed | Adaptive detector-driven synthetic 2D, RBF, top4/minprob0.50 | EfficientNetV2-S | **0.8149** | **0.4780** | **0.6806** | complete |
| Non-adaptive 2D baseline | MIP tri-view 256x384 | EfficientNetV2-S | 0.6175 | 0.1556 | 0.5008 | complete |
| Non-adaptive 2D baseline | MIP axial 256x384 | EfficientNetV2-S | 0.6075 | 0.1604 | 0.5061 | complete |
| Non-adaptive 2D baseline | Central axial slice 256x384 | EfficientNetV2-S | 0.6153 | 0.1610 | 0.4668 | complete |
| Guidance ablation | Fixed-control RBF, top4/minprob0.50 | EfficientNetV2-S | 0.5878 | 0.1109 | 0.4499 | complete |
| Guidance ablation | Random-control RBF, top4/minprob0.50 | EfficientNetV2-S | 0.6105 | 0.1877 | 0.5207 | complete |
| Volumetric baseline | 3D ResNet18, fit-pad 224x288x288 | ResNet18 | pending | pending | pending | running/to fill |

## Best current detector-driven synthetic result

La migliore configurazione completa e':

- detector: CPMNetv2 10-fold BF16
- working point: `threshold=0.50`, `top-k=4`
- interpolation: `RBF`
- classifier: `EfficientNetV2-S`
- evaluation: pooled over the 10 test folds
- support: `796` samples, `320` malignant, `476` benign

Risultato pooled:

| AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8149 | 0.4780 | 0.7513 | 0.6806 | 0.7033 | 0.6594 | 387 | 89 | 109 | 211 |

## Interpretation for paper

Le baseline non-adaptive mostrano cosa accade usando proiezioni o sezioni fisse, senza guida nodulare. Le MIP preservano informazione volumetrica globale, ma diluiscono il segnale nodulare: axial e tri-view restano intorno ad AUC 0.61 e MCC 0.16. La central axial slice e' una baseline ancora piu' minimale: nessuna aggregazione lungo l'asse z, solo la slice centrale del volume, e rimane su AUC 0.6153 e MCC 0.1610. Le ablation fixed/random mostrano che anche mantenendo RBF e training invariati, ma rimuovendo la guidance nodulare dai control points, la performance torna nello stesso intervallo delle baseline non-adaptive. Il miglior metodo adaptive RBF supera le baseline non-adaptive e le ablation di circa 0.20-0.23 AUC.

Questo supporta l'ipotesi centrale: non basta comprimere il volume in 2D; la proiezione deve essere guidata dalla localizzazione dei noduli e dal budget di falsi positivi. La rappresentazione adaptive consente al backbone 2D pretrained di lavorare su una superficie in cui il segnale clinicamente rilevante e' concentrato invece che disperso.

## Paper-ready deltas

| Comparison | Delta AUC | Delta MCC | Delta F1 |
|---|---:|---:|---:|
| Adaptive RBF top4/minprob0.50 vs MIP axial | +0.2074 | +0.3176 | +0.1745 |
| Adaptive RBF top4/minprob0.50 vs MIP tri-view | +0.1974 | +0.3224 | +0.1798 |
| Adaptive RBF top4/minprob0.50 vs central axial slice | +0.1996 | +0.3170 | +0.2138 |
| Adaptive RBF top4/minprob0.50 vs fixed-control RBF | +0.2271 | +0.3671 | +0.2307 |
| Adaptive RBF top4/minprob0.50 vs random-control RBF | +0.2044 | +0.2903 | +0.1599 |

## Notes

- Le metriche riportate come pooled sono ricalcolate concatenando le predizioni test di tutte le fold.
- La baseline 3D va completata appena finisce il run ResNet18; il placeholder e' gia' presente nella tabella master.
- I risultati GT synthetic sono utili come upper/reference bound, ma non devono essere presentati come confronto operativo diretto con detector-driven synthetic.
