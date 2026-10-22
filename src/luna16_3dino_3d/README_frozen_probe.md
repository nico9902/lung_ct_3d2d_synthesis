# Frozen 3DINO linear probes

`train_frozen_probe.py` runs two scan-level 3DINO probes on the existing ten
LUNA16 folds:

- `whole`: all overlapping 112³ crops, stride 56, from the processed volume.
- `detections`: up to four 112³ crops centred on out-of-fold CPMNetv2
  predictions with score at least 0.50. If none passes 0.50, use the single
  highest-scoring prediction. This fallback depends on detector scores only,
  never on the classification label or reference nodules.

In this cohort, 39 scans needed the below-threshold fallback and all 39 had
negative classification labels. Crop count also varies with the detector
threshold. The detector-guided result therefore cannot isolate 3DINO feature
quality from detector selection or the fallback pattern.

The official 3DINO teacher is frozen. Its normalized CLS embedding is computed
for each crop; per-scan mean and maximum embeddings are concatenated into a
2048-dimensional feature. The sole trainable layer is `Linear(2048, 1)`.
Feature extraction uses the same per-crop 0.05/99.95 percentile normalization
to [-1, 1] as the earlier partial-fine-tuning run. The `whole` variant reuses
that run's cached 112³ windows when available, or creates them from processed
CT volumes on a fresh run.

Head training follows the last partial-fine-tuning run: ten existing folds,
seed 233 + fold, weighted training sampler, effective batch 8, AdamW with
head learning rate 1e-4 and weight decay 1e-4, five warm-up epochs followed
by cosine decay, maximum 50 epochs, gradient clipping at 1.0, and patience 10.
The checkpoint is selected by **validation MCC** at score threshold 0.5.
The frozen encoder is evaluated in bfloat16 on CUDA; the linear head is trained
in float32.
The completed run used PyTorch 2.9.1+cu128 and torchvision 0.24.1+cu128,
compatible with the host's CUDA 12.8 driver.

First follow [foundation-model setup](../../FOUNDATION_MODELS.md). Export the
out-of-fold detector tables with `bash bash/cpmnetv2/export_synthesis_predictions.sh`;
this produces `outputs/cpmnetv2_paper_detector_all_oof_predictions.csv` with
`seriesuid`, `fold`, `x`, `y`, `z`, and `prob_nodule`. Every classification scan
must have at least one candidate for the detector-crop variant; missing scans
raise an error rather than changing the score-only fallback. Coordinates are
voxel positions on the processed grid.

Run both variants on a CUDA host:

```bash
python -m src.luna16_3dino_3d.train_frozen_probe --variant both \
  --checkpoint-path external/3DINO-ViT/3dino_vit_weights.pth
```

Feature extraction can be resumed after interruption: completed per-scan
features are skipped. For parallel extraction, run `--extract-only
--num-shards 2 --shard-index 0` and the matching command with shard index 1
on separate GPUs. When both finish, train with `--skip-extraction --variant
both`. Outputs are stored under `outputs/luna16_3dino_frozen_probe/{variant}`:
per-scan features, each fold's `best_val_mcc.pt`, history and test predictions,
and pooled out-of-fold test metrics.

Use `--code-dir` (or `DINO3D_CODE_DIR`), `--data-root`, `--splits-dir`,
`--detector-csv`, `--windows-cache-dir`, and `--output-dir` for other layouts.
Run `python -m src.luna16_3dino_3d.train_frozen_probe --help` for all options.
