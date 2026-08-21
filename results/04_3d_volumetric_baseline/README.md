# 3D Volumetric Baseline

Questa sezione e' dedicata alla baseline volumetrica:

`3D ResNet18 -> patient malignancy`

Configurazione prevista:

- input: LUNA16 preprocessed con spacing `1x1x1 mm`
- preprocessing: fit-pad aspect-preserving con zero padding
- target shape: `224x288x288`
- batch/accumulation: batch reale `4`, gradient accumulation `2`
- max epochs: `100`
- early stopping: disabilitato
- logging: W&B, incluse performance di test

Il README metodologico copiato da `docs/` spiega la scelta della dimensione e quantifica la perdita di informazione nodulare attesa.

La riga corrispondente nella tabella master resta `pending` finche' il run 3D non viene aggregato.
