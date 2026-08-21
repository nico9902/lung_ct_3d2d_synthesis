# LUNA16 2.5D Slice-Attention Baseline

Baseline preparata per confrontare Adaptive RBF con una rete 2.5D non-adaptive che usa tutte le slice assiali del volume preprocessato.

## Protocol

- representation: all axial slices from the LUNA16 preprocessed volume
- spacing: inherited from `LUNA16_preprocessed`, i.e. `1 x 1 x 1 mm`
- slice size: `256 x 384`
- slice encoder: `EfficientNetV2-S`, ImageNet pretrained
- aggregation: gated soft attention over slice-level features
- supervision: patient-level benign/malignant label
- folds: same LUNA16 10-fold splits
- metric protocol: pooled test metrics over the 10 folds
- optimization: AdamW, cosine annealing
- epochs: `100`
- batch size: `1`
- gradient accumulation: `8`
- precision: `16-mixed`
- checkpoint monitor: `val_mcc`

## Rationale

Questa baseline risponde a una domanda diversa rispetto a MIP e central slice: non comprime il volume in una singola proiezione fissa, ma consente a una backbone 2D pretrained di osservare tutte le slice assiali e di imparare quali pesare tramite attention. Rimane comunque non-adaptive, perche' non usa il detector per costruire una superficie o per concentrare la rappresentazione sulle regioni candidate nodulari.

E' quindi una baseline forte e scientificamente utile:

- se performa vicino ad Adaptive RBF, allora il vantaggio potrebbe dipendere soprattutto dall'uso di un encoder 2D pretrained e attention multi-slice;
- se resta sotto Adaptive RBF, il risultato supporta il valore specifico della superficie adaptive guidata dai noduli.

## Implementation

Codice:

- `src/luna16_slice_attention_2p5d/dataset.py`
- `src/luna16_slice_attention_2p5d/datamodule.py`
- `src/luna16_slice_attention_2p5d/model.py`
- `src/luna16_slice_attention_2p5d/lightning_model.py`
- `src/luna16_slice_attention_2p5d/train.py`
- `src/luna16_slice_attention_2p5d/aggregate.py`

Script server:

- `bash/luna16_slice_attention_2p5d/run_slice_attention_effnetv2s.sh`

Output remoto atteso:

- `outputs/luna16_slice_attention_2p5d_all_slices_256x384_effnetv2s`

## Results

Pending.
