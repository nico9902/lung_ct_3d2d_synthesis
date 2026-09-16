# LUNA16 Synthetic 2D Classifiers

Train PyTorch Lightning torchvision classifiers on SCPMNet TPS synthetic images.

The dataset is driven by:

- `data/LUNA16_preprocessed/cv_splits/luna16_classification_fold{fold}.csv`

The classification fold CSV provides scan IDs, `split`, `target`, and
`target_name`. Synthetic image paths are derived from each `seriesuid`; the
loader supports saliency GT outputs such as
`outputs/luna16_saliency_synthetic_gt/{seriesuid}/surface_grid_float_{seriesuid}.npy`
and TPS outputs such as
`outputs/scpmnet_luna16_10fold_tps_images/fold_{fold}/{seriesuid}_tps_top5.npy`.
Rows with `target_name=uncertain` are skipped when training with the default
binary classes `benign malignant`.

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

Freeze backbone layers:

```bash
python3 -m src.luna16_synthetic_2d.train --fold 0 --backbone densenet121 --freeze-half-backbone
python3 -m src.luna16_synthetic_2d.train --fold 0 --backbone resnet50 --freeze-first-layers 20
python3 -m src.luna16_synthetic_2d.train --fold 0 --backbone efficientnet_b0 --unfreeze-last-layers 12
python3 -m src.luna16_synthetic_2d.train --fold 0 --backbone efficientnet_b0 --freeze-backbone
```

`--freeze-half-backbone` freezes from the first backbone parameters until about
50% of the backbone scalar parameter count is frozen; the replaced classifier
head remains trainable.

Train an explicit split CSV:

```bash
python3 -m src.luna16_synthetic_2d.train \
  --synthetic-images-dir outputs/luna16_saliency_synthetic_gt \
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

The launcher also accepts `FREEZE_BACKBONE=1`, `FREEZE_HALF_BACKBONE=1`,
`FREEZE_FIRST_LAYERS=<N>`, or `UNFREEZE_LAST_LAYERS=<N>`.

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
