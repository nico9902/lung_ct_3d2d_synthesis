# LUNA16 2D Non-Adaptive Baselines - MIP EfficientNetV2-S

Risultati locali dei due esperimenti MIP:

- MIP assiale: stessa MIP assiale replicata su tre canali RGB.
- MIP tri-view: canale R assiale, canale G coronale, canale B sagittale.
- image size: `256x384`
- backbone: `EfficientNetV2-S`
- evaluation: pooled over the 10 test folds
- support: `796` samples, `320` malignant, `476` benign

## Pooled Results

| Method | AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MIP axial | 0.6075 | 0.1604 | 0.5930 | 0.5061 | 0.4940 | 0.5188 | 306 | 170 | 154 | 166 |
| MIP tri-view | 0.6175 | 0.1556 | 0.5917 | 0.5008 | 0.4924 | 0.5094 | 308 | 168 | 157 | 163 |

## Fold-wise Mean and Standard Deviation

| Method | AUC mean | AUC std | MCC mean | MCC std | F1 mean | F1 std |
|---|---:|---:|---:|---:|---:|---:|
| MIP axial | 0.6334 | 0.0706 | 0.1632 | 0.0963 | 0.4981 | 0.0857 |
| MIP tri-view | 0.6258 | 0.0529 | 0.1582 | 0.0979 | 0.4938 | 0.0675 |

## Comparison With Adaptive RBF

Current best adaptive detector-driven synthetic result:

- representation: adaptive synthetic surface
- detector working point: `top-k=4`, `threshold=0.50`
- interpolation: `RBF`
- backbone: `EfficientNetV2-S`
- AUC: `0.8149`
- MCC: `0.4780`
- F1: `0.6806`

| Comparison | Delta AUC | Delta MCC | Delta F1 |
|---|---:|---:|---:|
| Adaptive RBF vs MIP axial | +0.2074 | +0.3176 | +0.1745 |
| Adaptive RBF vs MIP tri-view | +0.1974 | +0.3224 | +0.1798 |

## Interpretation

MIP axial e MIP tri-view sono baseline non-adaptive forti da includere perche' usano direttamente il volume, ma comprimono l'informazione lungo assi fissi. In questi risultati il tri-view migliora solo marginalmente l'AUC rispetto all'assiale, mentre MCC e F1 restano sostanzialmente invariati.

Il gap con Adaptive RBF indica che il vantaggio non deriva semplicemente dal passare a una rappresentazione 2D, ma dall'usare una proiezione guidata dalla localizzazione nodulare.
