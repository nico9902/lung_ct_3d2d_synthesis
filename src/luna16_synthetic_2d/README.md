# LUNA16 Synthetic 2D Classifiers

Train PyTorch Lightning torchvision classifiers on SCPMNet TPS synthetic images.

The dataset is driven by:

- `outputs/scpmnet_luna16_10fold_tps_images/manifest.csv`
- `data/LUNA16_preprocessed/cv_splits/luna16_classification_fold{fold}.csv`

The manifest provides the synthetic image path, and the classification fold CSV
provides `split`, `target`, and `target_name`. Rows with `target_name=uncertain`
are skipped when training with the default binary classes `benign malignant`.

## Files

- `dataset.py`: one fold-CSV-driven `SyntheticLuna16Dataset`
- `datamodule.py`: Lightning `DataModule` and transforms
- `models.py`: torchvision backbone factory
- `lightning_model.py`: Lightning classifier
- `train.py`: CLI entry point

## Examples

Train fold 0:

```bash
python3 -m src.luna16_synthetic_2d.train \
  --fold 0 \
  --backbone densenet121 \
  --image-size 256 384 \
  --epochs 50 \
  --batch-size 16 \
  --precision 16-mixed
```

Train an explicit split CSV:

```bash
python3 -m src.luna16_synthetic_2d.train \
  --manifest-csv outputs/scpmnet_luna16_10fold_tps_images/manifest.csv \
  --split-csv data/LUNA16_preprocessed/cv_splits/luna16_classification_fold0.csv \
  --backbone resnet50
```

Run the default backbone set:

```bash
src/luna16_synthetic_2d/run_backbones.sh
```

Run a subset:

```bash
FOLDS="0 1" BACKBONES="resnet18 densenet121" src/luna16_synthetic_2d/run_backbones.sh
```

Outputs are written to `outputs/luna16_synthetic_2d/fold_<fold>/<backbone>/`:

- `checkpoints/*.ckpt`: best and last Lightning checkpoints
- `version_*/metrics.csv`: Lightning CSV logs
- `config.json`: run configuration

Evaluate an existing checkpoint:

```bash
python3 -m src.luna16_synthetic_2d.train \
  --fold 0 \
  --backbone densenet121 \
  --eval-only outputs/luna16_synthetic_2d/fold_0/densenet121/checkpoints/last.ckpt
```
