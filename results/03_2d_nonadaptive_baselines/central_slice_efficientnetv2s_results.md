# LUNA16 2D Non-Adaptive Baseline - Central Axial Slice EfficientNetV2-S

Baseline completa su 10 fold.

Configurazione:

- representation: central axial slice
- slice selection: `z = D // 2` sul volume LUNA16 preprocessed
- channels: stessa slice replicata su RGB
- image size: `256x384`
- backbone: `EfficientNetV2-S`
- folds: `0..9`
- epochs: `100`
- batch size: `16`
- precision: `32`
- W&B project: `luna16-2d-baselines-nonadaptive`

Output remoto atteso:

- dataset: `/ssd2/domenico/datasets/2d_baselines/luna16_central_slice_central_axial_256x384`
- training output: `outputs/luna16_2d_baseline_central_slice_central_axial_256x384_efficientnet_v2_s`
- nohup log: `logs/luna16_2d_baselines/central_axial_efficientnetv2s.nohup.log`

## Rationale

La central axial slice e' una baseline non-adaptive minimale: non usa il detector, non usa annotazioni nodulari, non costruisce una superficie sintetica e non aggrega informazione lungo l'asse z come la MIP. Serve a misurare quanto del segnale classificativo sia catturabile da una singola sezione fissa del volume.

Questa baseline e' volutamente sfavorevole rispetto alle rappresentazioni adaptive, perche' i noduli possono trovarsi lontano dalla slice centrale. Proprio per questo e' utile nel paper: dimostra che una semplice riduzione 3D-to-2D non basta, e che la scelta del piano/superficie deve essere guidata dal contenuto nodulare.

## Results

### Pooled Test Performance

Metriche calcolate aggregando le predizioni di test di tutti i 10 fold, sullo stesso insieme di 796 pazienti usato per gli altri confronti.

| Representation | Backbone | Samples | Positives | Negatives | AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Central axial slice | EfficientNetV2-S | 796 | 320 | 476 | 0.6153 | 0.1610 | 0.6068 | 0.4668 | 0.5131 | 0.4281 | 346 | 130 | 183 | 137 |

### Per-Fold Test Performance

| Fold | AUC | MCC | Accuracy | F1 |
|---:|---:|---:|---:|---:|
| 0 | 0.6624 | 0.1089 | 0.5570 | 0.4068 |
| 1 | 0.6803 | 0.1939 | 0.6203 | 0.5000 |
| 2 | 0.6483 | 0.2534 | 0.6627 | 0.4815 |
| 3 | 0.6667 | 0.1795 | 0.5949 | 0.4839 |
| 4 | 0.6128 | 0.1706 | 0.6026 | 0.4918 |
| 5 | 0.6089 | 0.1232 | 0.5823 | 0.4407 |
| 6 | 0.5466 | 0.0176 | 0.5679 | 0.2857 |
| 7 | 0.6580 | 0.2486 | 0.6364 | 0.5172 |
| 8 | 0.6352 | 0.2082 | 0.6173 | 0.4746 |
| 9 | 0.6312 | 0.2263 | 0.6250 | 0.5455 |
| Mean ± std_pop | 0.6350 ± 0.0368 | 0.1730 ± 0.0689 | 0.6066 ± 0.0306 | 0.4628 ± 0.0692 |

### Interpretation

La central axial slice e' chiaramente inferiore alla Adaptive RBF detector-guided con EfficientNetV2-S, che raggiunge AUC pooled `0.8149` e MCC pooled `0.4780`. Il gap supporta l'idea che una singola slice fissa non conservi abbastanza informazione nodulare e che la guidance spaziale del detector sia una componente sostanziale del metodo.
