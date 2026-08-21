# RBF Control-Point Guidance Ablation

Esperimento completo per dimostrare che la guidance spaziale del detector e' fondamentale nella rappresentazione adaptive.

## Reference

Configurazione principale gia' completata:

- representation: detector-guided adaptive RBF surface
- detector working point: `top-k=4`, `threshold=0.50`
- interpolation: `RBF`
- control points: detector candidates + contour points
- backbone: `EfficientNetV2-S`
- pooled AUC: `0.8149`
- pooled MCC: `0.4780`
- pooled F1: `0.6806`

## Ablations

Le due ablation mantengono invariato tutto il resto del pipeline:

- stessi pazienti e stessi split 10-fold
- stesso detector usato solo per stabilire quali scans hanno candidati dopo threshold
- stesso `top-k=4`
- stesso numero di punti per candidato: centro + 4 contour points
- stesso RBF smoothing, boundary anchors e lung-volume anchors
- stesso backbone `EfficientNetV2-S`
- stesso training delle backbone 2D
- stesse metriche pooled

Cambiano solo i control points.

### Fixed Control Points

Sostituisce i control points detector-guided con punti anatomici fissi e uguali per tutti i pazienti:

- quattro posizioni normalizzate nel piano assiale: quadranti centro-sinistra/destra e superiore/inferiore
- z fissata alla meta' del volume
- raggi derivati dalla mediana dei raggi dei candidati detector selezionati, per mantenere una scala comparabile

Scopo: testare se basta usare una superficie RBF regolare e sempre uguale, senza informazione nodulare.

### Random Control Points

Sostituisce i control points detector-guided con punti casuali riproducibili:

- seed stabile per paziente
- punti campionati nella lung mask quando disponibile
- raggi derivati dai candidati detector selezionati

Scopo: testare se basta usare superfici RBF variabili e plausibili nel polmone, ma non guidate dalla posizione dei noduli.

## Expected Interpretation

Se fixed/random performano sensibilmente peggio del detector-guided RBF, il risultato supporta una conclusione forte:

la performance non dipende semplicemente da RBF, dal backbone 2D pretrained o dalla compressione 3D-to-2D, ma dalla scelta guidata dei control points in regioni nodulari candidate.

## Scripts

Generazione sintetiche:

- `bash/run_luna16_detector_saliency_top4_minprob0.5_rbf_fixed.sh`
- `bash/run_luna16_detector_saliency_top4_minprob0.5_rbf_random.sh`
- `bash/run_luna16_detector_saliency_top4_minprob0.5_rbf_control_ablation.sh`

Training backbone:

- `src/luna16_synthetic_2d/run_backbones_det_top4_minprob0.5_rbf_fixed_control.sh`
- `src/luna16_synthetic_2d/run_backbones_det_top4_minprob0.5_rbf_random_control.sh`
- `src/luna16_synthetic_2d/run_backbones_det_top4_minprob0.5_rbf_control_ablation.sh`

## Result Table

Metriche pooled calcolate concatenando le predizioni di test dei 10 fold.

| Method | Backbone | Samples | AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Detector-guided RBF | EfficientNetV2-S | 796 | 0.8149 | 0.4780 | 0.7513 | 0.6806 | 0.7033 | 0.6594 | 387 | 89 | 109 | 211 | complete |
| Fixed-control RBF | EfficientNetV2-S | 796 | 0.5878 | 0.1109 | 0.5791 | 0.4499 | 0.4740 | 0.4281 | 324 | 152 | 183 | 137 | complete |
| Random-control RBF | EfficientNetV2-S | 796 | 0.6105 | 0.1877 | 0.6068 | 0.5207 | 0.5105 | 0.5312 | 313 | 163 | 150 | 170 | complete |

## Delta vs Detector-Guided RBF

| Comparison | Delta AUC | Delta MCC | Delta F1 |
|---|---:|---:|---:|
| Detector-guided RBF vs fixed-control RBF | +0.2271 | +0.3671 | +0.2307 |
| Detector-guided RBF vs random-control RBF | +0.2044 | +0.2903 | +0.1599 |

## Interpretation

Entrambe le ablation collassano verso le baseline non-adaptive: fixed-control raggiunge AUC `0.5878`, random-control AUC `0.6105`, mentre la RBF guidata dal detector arriva a AUC `0.8149`. Questo supporta una conclusione importante per il paper: il vantaggio non deriva semplicemente dall'uso di una superficie RBF o dal backbone 2D pretrained, ma dal posizionamento dei control points sulle regioni candidate nodulari.
