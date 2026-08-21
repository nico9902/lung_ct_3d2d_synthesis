# Detection-Crop MIL Baseline

Baseline proposta per rispondere alla domanda del revisore:

> Se il detector ha gia' trovato i noduli, perche' costruire una superficie RBF/TPS e sintetizzare un'immagine 2D? Perche' non classificare direttamente i noduli/candidati e aggregarne le predizioni?

## Protocol

- detector: CPMNetv2 10-fold BF16, stesse prediction usate per Adaptive RBF
- working point: `top-k=4`, `threshold=0.50`
- instances: crop assiali 2D centrati sui candidati detector
- crop physical size: `64 x 64 mm`
- crop image size: `224 x 224`
- encoder: `EfficientNetV2-S`, ImageNet pretrained
- patient aggregation: mean pooling, max pooling, gated/soft attention
- supervision: solo label paziente benign/malignant
- split: stessi 10 fold LUNA16, stesse metriche pooled
- checkpoint: `val_mcc`
- W&B project: `luna16-detection-mil`

## Scientific Rationale

Questa baseline mantiene il vantaggio informativo del detector ma rimuove il contributo specifico della rappresentazione adaptive. Se la MIL sui crop raggiungesse performance simili alla Adaptive RBF, il miglioramento potrebbe essere spiegato solo dalla qualita' delle detections. Se invece rimane inferiore, il risultato supporta l'idea che la superficie sintetica aggiunga informazione utile: contesto anatomico, organizzazione spaziale dei candidati e una rappresentazione paziente-level piu' coerente per un backbone 2D pretrained.

## Output

Output remoto atteso:

- `outputs/luna16_detection_mil_cpmnetv2_top4_minprob0.50_effnetv2s`

Script:

- `bash/luna16_detection_mil/run_detection_mil_efficientnetv2s.sh`

## Results

Pending.
