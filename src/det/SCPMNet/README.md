# SCPM-Net

This folder contains a PyTorch Lightning implementation of **SCPM-Net: an anchor-free 3D lung nodule detector using sphere representation and center-points matching**.

It uses the released SCPM-Net backbone from the paper and adds the missing training code:

- 3D ResNet/FPN backbone with squeeze-excitation channel attention.
- Multi-level spatial coordinate maps concatenated into the feature pyramid.
- Two detection heads that predict center logits, sphere radius, and local center offset.
- Anchor-free center point target assignment.
- Focal/OHEM-style classification loss for class imbalance.
- Radius, offset, and sphere DIoU/SIoU-style geometric loss.
- PyTorch Lightning training, validation, checkpointing, and test prediction export.
- Hydra config and bash launcher.

## Files

- `model.py`: SCPM-Net backbone and heads.
- `losses.py`: center-point matching target builder and sphere loss.
- `dataset.py`: CSV-driven 3D CT volume dataset.
- `datamodule.py`: Lightning dataloaders.
- `lightning_model.py`: training/test logic and detection decoding.
- `train_lightning.py`: Hydra entrypoint.
- `conf/train_lightning.yaml`: default training configuration.
- `../../../bash/scpmnet_train_lightning_hydra.sh`: shell launcher.

## Annotation CSV

The default dataset now points to `data/lidc_process/lidc_labels.csv`. That CSV is expected to contain:

```text
seriesuid,image_path,split,x,y,z,w,h,d,label
```

Use `split` values `train`, `val`, and `test`. `image_path` may be absolute or relative to `data_root`. LIDC NIfTI volumes load as `x, y, z`, so the loader transposes them to the detector order `z, y, x`. The LIDC columns `x, y, z` are also converted internally to `z, y, x`, so image voxels and targets use the same coordinate system. Because `lidc_labels.csv` has `w, h, d` instead of diameter, the loader represents each nodule as a sphere with radius:

```text
radius = max(w, h, d) / 2
```

Rows with a `label` column are used as positives only when the label is `nodule`, `1`, `true`, or `positive`.

By default, `skip_missing_images: true` filters out CSV rows whose `image_path` does not exist. This is useful with a partial local `lidc_process` folder. Set it to `false` only when every referenced series exists in your dataset root.

The loader also still accepts generic sphere labels with `coordZ, coordY, coordX, radius`. If your labels use diameter, replace `radius` with `diameter` or `diameter_mm`; the loader divides it by two.

For mask-based labels, add a column containing the mask path and set:

```bash
mask_path_column=nodule_mask_path
```

The loader extracts connected components and converts them to center/radius sphere annotations.

Supported image formats are `.nii`, `.nii.gz`, and `.npy`.

## Subvolume Sampling

SCPM-Net uses fixed 3D subvolumes instead of loading an entire CT scan into the network. The subvolume size is controlled by:

```yaml
crop_size:
  - 96
  - 96
  - 96
```

For each `seriesuid`, all annotation rows are grouped together. A sampled crop keeps only the nodules whose centers fall inside that crop, and their coordinates are shifted from scan coordinates into crop-local coordinates.

### Training

Training uses random subvolume sampling.

- `samples_per_volume` controls how many crops each scan contributes per epoch.
- With probability `positive_crop_prob`, the crop is sampled around a randomly selected nodule center.
- Positive crop centers are jittered by up to 25% of the crop size along each axis.
- Otherwise, the crop is sampled from a random valid location in the scan.
- If a crop touches the scan boundary, missing voxels are padded with `-1`.

The paper-style config uses:

```yaml
crop_size: [96, 96, 96]
samples_per_volume: 1
positive_crop_prob: 0.7
```

### Validation

Validation currently uses the same fixed-size crop dataset as training, but the DataLoader uses `shuffle=False`.

So validation loss is computed on sampled 96³ subvolumes, not exhaustive full-scan tiling. This is useful for monitoring optimization, but it is not the same as paper-style scan-level FROC evaluation.

### Test

Test uses full-volume sliding-window inference by default:

```yaml
test_full_volume: true
sliding_window_stride: [24, 24, 24]
```

For each scan in the `test` split, the dataset tiles the full volume with 96³ crops. The last crop on each axis is forced to touch the scan boundary, so the whole scan is covered even when the size is not divisible by the stride.

During test:

- each crop is passed through SCPM-Net independently;
- crop-local detections are shifted back into global scan coordinates using the crop origin;
- detections from all windows of the same scan are merged;
- sphere NMS is applied scan-by-scan;
- the top `final_topk` detections per scan are kept.

Decoded full-scan detections are written to:

```text
outputs/scpmnet/<experiment_name>/predictions/test_predictions.csv
```

with global voxel-space columns:

```text
seriesuid,coordZ,coordY,coordX,radius,probability
```

FROC is computed automatically when `evaluate_froc: true`. The evaluator uses LUNA-style center-distance matching by default: a prediction is a true positive when its center falls within the ground-truth nodule radius and that nodule has not already been matched by a higher-confidence prediction. Sensitivity is reported at:

```text
0.125, 0.25, 0.5, 1, 2, 4, 8 FP/scan
```

and the mean of these seven sensitivities is logged as `test/mean_froc`. FROC files are written next to predictions:

```text
test_froc.csv
test_froc_curve.csv
```

To use sphere-IoU matching instead, set:

```yaml
froc_match_strategy: sphere_iou
froc_iou_threshold: 0.1
```

Set `test_full_volume: false` only if you want the older crop-sampled test behavior for debugging.

## Training

From the repository root:

```bash
CSV_PATH=data/lidc_process/lidc_labels.csv \
DATA_ROOT=. \
bash bash/scpmnet_train_lightning_hydra.sh
```

Or call Hydra directly:

```bash
python -m src.det.SCPMNet.train_lightning \
  csv_path=data/lidc_process/lidc_labels.csv \
  data_root=. \
  batch_size=2 \
  devices="[0]" \
  precision=16-mixed
```

To use the paper-style hyperparameter preset:

```bash
CSV_PATH=data/lidc_process/lidc_labels.csv \
DATA_ROOT=. \
bash bash/scpmnet_train_paper_hydra.sh
```

The paper preset lives in `conf/train_lightning_paper.yaml` and uses 96³ crops, 170 epochs, SGD with momentum 0.9, weight decay 1e-4, batch size 24, the 20-epoch warmup/milestone LR schedule, K=7 center points, OHEM n=100, and the reported re-focal/radius/sphere-loss settings.

Checkpoints are written to:

```text
outputs/scpmnet/scpmnet/checkpoints/
```

## Test Only

```bash
python -m src.det.SCPMNet.train_lightning \
  test_only=true \
  checkpoint=outputs/scpmnet/scpmnet/checkpoints/last.ckpt
```

Test predictions are saved as:

```text
outputs/scpmnet/scpmnet/predictions/test_predictions.csv
```

with columns:

```text
seriesuid,coordZ,coordY,coordX,radius,probability
```

## Important Notes

The paper reports LUNA16 performance after its full preprocessing and evaluation setup. This implementation recreates the SCPM-Net architecture and training objective from the supplied paper/code, but you still need consistent resampling, train/val/test splits, and LUNA-style FROC evaluation if you want paper-comparable numbers.
