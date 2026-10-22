# Adaptive 2D synthesis and classification

CPMNetv2 out-of-fold candidates guide an RBF or Shepard surface through each
processed CT. The resulting three-channel image is classified with torchvision
backbones. The paper's primary configuration is RBF, top-4 candidates with
probability >= 0.50, and EfficientNetV2-S.

## Inputs and generation

Run from the repository root after installing `requirements.txt`. Preprocessed
CTs and masks live in `data/processed`; patient classification CSVs live in
`data/processed/cv_splits/luna16_classification_fold{fold}.csv`. CSVs provide
`seriesuid`, `split`, `target`, and `target_name`; uncertain cases are excluded.
See the [main README](../../README.md) for rebuilding labels and folds.

After detector training, export its box predictions before synthesizing images:

```bash
bash bash/cpmnetv2/export_synthesis_predictions.sh
bash bash/luna16_synthetic_2d/generate_top4_minprob0.5_rbf.sh
```

`DETECTOR_PREDICTIONS` selects the exported candidate root, `DATA_ROOT` the
processed volumes, and `SYNTHETIC_ROOT` the output parent. The default RBF image
root is `data/synthetic_2d/luna16_saliency_synthetic_detector_cpmnetv2_bf16_top4_minprob0.50_rbf`.
The classifier uses `surface_<seriesuid>.png` in each scan directory.
`surface_grid_float_<seriesuid>.npy` stores sampling depths for explanation
analysis; retain the PNG files to avoid the loader's legacy grid fallback.
Legacy TPS filename support remains for previously generated datasets.

## Training and evaluation

Run one fold of the primary experiment:

```bash
FOLDS=0 BACKBONES=efficientnet_v2_s bash bash/luna16_synthetic_2d/train_top4_minprob0.5_rbf.sh
```

Omit `FOLDS=0` to run all ten folds. `BACKBONES` can be a space-separated list.
The [launcher guide](../../bash/README.md) lists Shepard, geometric controls,
MIP, detector-selected slice and volumetric experiments.

For direct CLI use, specify both data locations:

```bash
python -m src.luna16_synthetic_2d.train \
  --synthetic-images-dir data/synthetic_2d/luna16_saliency_synthetic_detector_cpmnetv2_bf16_top4_minprob0.50_rbf \
  --splits-dir data/processed/cv_splits \
  --fold 0 --backbone efficientnet_v2_s --image-size 256 384
```

Add `--eval-only /path/to/best.ckpt` to evaluate a saved checkpoint with the
same input configuration. Freezing is controlled by `--freeze-backbone`,
`--freeze-half-backbone`, `--freeze-first-layers N`, or `--unfreeze-last-layers N`.
The launchers expose corresponding uppercase environment variables.

`train_top4_minprob0.5_rbf.sh` writes checkpoints, run configuration and test
predictions under `outputs/luna16_synthetic_2d_cpmnetv2_bf16_top4_minprob0.50_rbf_v100`;
`OUTPUT_DIR` overrides this root. It also aggregates fold predictions.

## Explanations and statistical comparisons

```bash
bash bash/luna16_synthetic_2d/gradcam_top4_minprob0.5_rbf.sh
python -m src.luna16_synthetic_2d.gradcam_nodule_energy --help
python -m src.luna16_synthetic_2d.manuscript_paired_comparisons --help
```

The energy analysis measures the fraction of Grad-CAM energy within projected
and surface-sampled reference-nodule masks. It needs trained checkpoints,
synthetic images and saved sampling surfaces, processed nodule masks, label
CSV and the ten classification splits. Override `--run-root`,
`--synthetic-images-dir`, `--processed-root`, `--labels-csv`, and `--splits-dir`
as needed. Outputs go to `outputs/luna16_synthetic_2d_gradcam_nodule_energy`.

Paired comparisons compute DeLong AUC and exact McNemar tests, followed by
Benjamini-Hochberg adjustment across the chosen comparator family. Without
`--config`, the script uses the original local experiment directory names.
For a new checkout, provide a JSON manifest with paths relative to that file:

```json
{
  "reference": {"path": "outputs/rbf/all_test_predictions.csv", "backbone": "efficientnet_v2_s"},
  "comparators": [
    {"name": "Adaptive Shepard", "path": "outputs/shepard/all_test_predictions.csv", "backbone": "efficientnet_v2_s"}
  ]
}
```

```bash
python -m src.luna16_synthetic_2d.manuscript_paired_comparisons \
  --config comparisons.json --output outputs/manuscript_paired_comparisons.csv
```

Each source can be a pooled CSV or a directory of fold prediction files.
Required columns are `sample_id` (or `seriesuid`), `fold`, `label`, `prediction`,
and `score`; `backbone` is needed only when filtering by backbone. Each source
must contain the same 796 patients across ten test folds. Include the complete
planned comparator family when reproducing adjusted p-values.
