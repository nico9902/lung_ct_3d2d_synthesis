# Detection-Guided Adaptive 2D Synthesis for Lung CT

This repository contains the code and experiment documentation for a lung CT representation-learning project focused on patient-level malignancy assessment from LUNA16/LIDC-IDRI volumes.

The central idea is to use a 3D pulmonary nodule detector to guide the synthesis of a compact 2D image. Detector candidates are converted into geometric control regions, an adaptive surface is fitted through the CT volume, and the resulting synthetic 2D representation is classified with standard 2D backbones.

## Method Overview

The proposed pipeline is:

1. Preprocess CT volumes to isotropic 1 mm spacing and lung-focused crops.
2. Run a 3D nodule detector, currently CPMNetv2/SCPM-Net style center-points matching.
3. Select a detector working point using a representation-budget criterion.
4. Generate adaptive 2D synthetic images using:
   - RBF interpolation;
   - Shepard interpolation as an adaptive comparator;
   - fixed/random control-point ablations.
5. Train patient-level 2D classifiers on the synthesized images.
6. Compare against non-adaptive 2D baselines, detector crop-MIL, and a 3D ResNet18 volumetric baseline.

## Repository Layout

```text
bash/                         Experiment launch scripts
results/                      Curated result summaries for paper writing
src/det/CPMNetv2/             Detector code used for nodule localization
src/luna16_synthetic_2d/      Adaptive 2D synthesis and 2D backbone experiments
src/luna16_detection_mil/     Detector-crop MIL baseline
src/luna16_volume_3d/         3D ResNet18 volumetric baseline
src/luna16_colipri_3d/        Frozen COLIPRI-CRM foundation-model linear-probe baseline
src/luna16_radjepa_3d/        Frozen Rad-JEPA-3D foundation-model linear-probe baseline
src/luna16_3dino_3d/          Partial-fine-tuning 3DINO-ViT foundation-model baseline
src/prs/                      Preprocessing and label-generation utilities
```

Large local folders such as `data/`, `outputs/`, `docs/`, `others/`, `latex/`, `wandb/`, `wandb2/`, model checkpoints, medical images, and NumPy arrays are intentionally ignored by Git.

## Main Results

Curated paper-ready summaries are organized under [`results/`](results/):

- [`results/00_paper_overview`](results/00_paper_overview): master result tables.
- [`results/01_synthetic_2d_adaptive`](results/01_synthetic_2d_adaptive): adaptive RBF/Shepard and control-point ablations.
- [`results/02_detector_and_working_points`](results/02_detector_and_working_points): detector FROC and working-point analysis.
- [`results/03_2d_nonadaptive_baselines`](results/03_2d_nonadaptive_baselines): MIP, central slice, and crop-MIL baselines.
- [`results/04_3d_volumetric_baseline`](results/04_3d_volumetric_baseline): 3D ResNet18 baseline.
- [`results/08_colipri_foundation_baseline`](results/08_colipri_foundation_baseline): frozen COLIPRI-CRM foundation-model linear-probe baseline.
- [`results/09_radjepa_foundation_baseline`](results/09_radjepa_foundation_baseline): frozen Rad-JEPA-3D foundation-model linear-probe baseline.
- [`results/10_3dino_foundation_baseline`](results/10_3dino_foundation_baseline): partial-fine-tuning 3DINO-ViT foundation-model baseline.
- [`results/06_statistical_tests`](results/06_statistical_tests): paired statistical comparisons.
- [`results/07_computational_analysis`](results/07_computational_analysis): input complexity, FLOPs, and runtime notes.

The main detector-guided RBF configuration uses CPMNetv2 detections with `threshold = 0.50` and `top-k = 4`. With EfficientNetV2-S, the pooled patient-level performance is:

```text
AUC  = 0.8149
MCC  = 0.4780
F1   = 0.6806
ACC  = 0.7513
```

## Installation

Create a Python environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Some experiments require additional detector-specific dependencies under `src/det/` and GPU-enabled PyTorch. The exact environment may need to be adapted to the target CUDA version.

## Data

The code expects preprocessed LUNA16/LIDC-IDRI data locally, but the raw and preprocessed medical images are not distributed in this repository.

Typical local paths used during development were:

```text
data/LUNA16/
data/LUNA16_preprocessed/
data/LIDC-IDRI files/
outputs/
```

## Grad-CAM Explanations

After training a 2D synthetic-image classifier, Grad-CAM figures can be exported on the server with:

```bash
python -m src.luna16_synthetic_2d.explain_gradcam \
  --checkpoint outputs/luna16_synthetic_2d_top4_minprob0.5_rbf/fold_0/efficientnet_v2_s/checkpoints/best.ckpt \
  --synthetic-images-dir data/luna16_saliency_synthetic_detector_top4_minprob0.5_rbf \
  --split-csv data/LUNA16_preprocessed/cv_splits/luna16_classification_fold0.csv \
  --fold 0 \
  --backbone efficientnet_v2_s \
  --processed-dir data/LUNA16_preprocessed \
  --clip-gt-to-lung \
  --exclude-sample-ids 120188208010880 \
  --selection balanced_highest_score \
  --target-class predicted \
  --require-malignant-gt-hit \
  --require-gt-inside-visible-lung \
  --figure-layout paper_overlay \
  --max-samples 24 \
  --output-dir outputs/luna16_synthetic_2d_gradcam/fold_0_efficientnet_v2_s
```

The script writes per-sample input, ground-truth nodule-mask overlay when available, heatmap, overlay, paper-style multi-panel figures, a class-balanced `gradcam_summary.png`, a `gradcam_manifest.csv`, and a markdown report. GT nodule masks are clipped to the available lung mask by default to avoid displaying annotations outside the lung. Balanced selections keep equal class counts; if one fold has too few malignant or benign cases, fewer than `--max-samples` examples are exported unless `--balance-fill-shortfall` is passed. With `--require-malignant-gt-hit`, malignant examples are kept only when the Grad-CAM hotspot overlaps the clipped ground-truth nodule mask. With `--require-gt-inside-visible-lung`, samples whose projected GT mask falls outside the visible lung silhouette are discarded.

## COLIPRI Foundation-Model Baseline

`src/luna16_colipri_3d/` compares the Adaptive RBF approach against a **frozen** [COLIPRI-CRM](https://huggingface.co/microsoft/colipri) 3D CT foundation-model encoder, using the *exact same* patient labels and 10-fold splits as the rest of the project (`data/processed/cv_splits/luna16_classification_fold{0-9}.csv`; every fold CSV already contains the same 888 patients/labels, only the `split` column differs, so no new folds are created and there is no patient leakage beyond the LUNA16-subset grouping already baked into those folds).

Pipeline:

1. **Feature extraction** (`extract_features.py`) loads the official `colipri` package (`pip install colipri torchio`) and runs the official `Processor` directly on the **raw** LUNA16 volumes (`data/raw/LUNA16/subsets/subsetN/subsetN/<seriesuid>.mhd`, native spacing/HU) — not the repo's lung-windowed 8-bit preprocessed volumes, since COLIPRI's own processor performs reorientation, resampling to 2 mm isotropic, HU clamping/rescaling, and center-crop/pad to 192x192x192 itself. The frozen encoder produces, per patient:
   - `pooled`: the official pooled+projected embedding (768-D);
   - `dense_maxpool`: the dense projected feature grid (768, 24, 24, 24) reduced with a channel-wise global max over the spatial grid (768-D).

   Embeddings are cached per patient under `outputs/luna16_colipri_3d/embeddings/per_patient/<seriesuid>.npz` (safe to interrupt/resume) and consolidated into `outputs/luna16_colipri_3d/embeddings/colipri_embeddings.npz`, so the ~150 GB of raw CT volumes only need to be read once regardless of how many folds/representations are probed afterwards.

2. **Linear probing** (`train_linear_probe.py`) trains a single frozen `Linear(768, 1)` head per fold/representation with `BCEWithLogitsLoss` (class-balanced via `pos_weight`), AdamW + cosine LR, and validation-based early stopping/checkpoint selection (`--monitor val_mcc` by default) — mirroring the training-loop and output-file conventions (`config.json`, `history.csv`, `best.pt`, `test_metrics.json`, `test_predictions.csv`) already used by `src/luna16_volume_3d/train_resnet18.py`.

3. **Aggregation** (`aggregate.py`) pools all 10 folds' out-of-fold test predictions per representation into a single AUC/MCC/F1/accuracy/precision/recall/confusion-matrix report, and separately reports the per-fold mean +/- std of each metric.

Run the full baseline (feature extraction once, then 10 folds x 2 representations, then aggregation) with:

```bash
CUDA_VISIBLE_DEVICES=3 bash bash/luna16_colipri_3d/run_colipri_baseline.sh
```

(`CUDA_VISIBLE_DEVICES=3` selects the A100 on this host; since it is the only visible device, PyTorch addresses it as `cuda:0`. The script also sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` internally — without it, PyTorch's default CUDA device enumeration does **not** match `nvidia-smi`'s indices on this host, and index `3` silently resolves to a V100 instead of the A100.)

Results are written to:

- `outputs/luna16_colipri_3d/embeddings/` — cached embeddings, `colipri_embeddings_index.csv`, and `extraction_config.json` (checkpoint/processor/environment info for reproducibility).
- `outputs/luna16_colipri_3d/probes/fold_<N>/<pooled|dense_maxpool>/` — per-fold probe checkpoints, training history, test metrics, and per-patient out-of-fold predictions.
- `outputs/luna16_colipri_3d/probes/{pooled_metrics.csv,per_fold_mean_std_metrics.csv,colipri_pooled_results.md}` — cross-fold pooled and mean/std reports.
- `results/08_colipri_foundation_baseline/` — the curated copy of the final report, for paper writing alongside the other baselines under `results/`.

## Rad-JEPA-3D Foundation-Model Baseline

`src/luna16_radjepa_3d/` compares against a second **frozen** 3D foundation model, Rad-JEPA-3D (an H-Mamba hybrid
encoder), on the exact same 10 LUNA16 folds. Unlike COLIPRI's official pip package, Rad-JEPA-3D's checkpoint and
code are not published as an installable package; this baseline reuses the checkpoint and encoder code already
present in the sibling project `/home/domenico/lung-jepa-world-model` on this host, by absolute path, rather than
re-downloading or re-implementing the architecture. No lesion-guided information (CPMNetv2 detections, nodule
coordinates, masks) is used — Rad-JEPA only ever sees the full CT volume, so it represents a generic full-volume
foundation-model baseline.

Rad-JEPA-3D expects volumes shaped `(1, 32, 256, 256)`. Four patient-level representation strategies are compared,
all derived from the *same* per-window cached embeddings (the encoder is never re-run to compare strategies):

1. **resize32**: the whole scan resized (depth included) to `(32, 256, 256)` — one embedding per patient.
2. **sliding_mean**: 32-slice windows with stride 16 over the native-depth, in-plane-256x256-resized scan; the
   patient representation is the element-wise mean of the per-window embeddings.
3. **sliding_max**: same windows, element-wise max instead of mean.
4. **sliding_mean_max**: concatenation of the mean- and max-pooled window embeddings.

Every window's embedding is the encoder's mean-pooled token output — `encoder(x, indices=None).mean(dim=1)` →
384-D — the exact convention already used by the sibling project's own extraction script (verified against the
actual `MambaEncoder.forward` source rather than assumed): the frozen `target_encoder` weights load with 0
missing/0 unexpected keys and this pooling matches how that project already extracts scan-level features.

Pipeline:

1. **Feature extraction** (`extract_features.py`) reuses `find_raw_ct_path`/`load_patient_table` from the COLIPRI
   baseline (same raw LUNA16 `.mhd` source, same 888-patient/label invariant across folds) and HU-windows
   ([-1000, 400] → [0, 1], the sibling project's established convention) each raw volume before resizing. It
   caches, per patient, the `resize32` embedding plus every sliding-window embedding with its `(window_start,
   window_end)` — window embeddings are not duplicated per fold, since they don't depend on the fold split.
   **This step must run with the sibling project's Python** (`/home/domenico/lung-jepa-world-model/.venv/bin/python3`),
   which has the `mamba_ssm`/`causal_conv1d` CUDA extensions the encoder depends on; this repo's own `myenv` is
   never modified for this baseline (its `torch==2.10.0` is incompatible with Rad-JEPA's required
   `torch==2.5.1+cu121`).
2. **Linear probing** (`train_linear_probe.py`) derives one of the four representations above from the cache and
   trains a `Linear(feature_dim, 1)` head — reusing the exact training loop, metrics, early-stopping and
   output-file conventions from the COLIPRI baseline's probe script (imported directly, not duplicated). Runs with
   this repo's normal `myenv` (no Rad-JEPA-specific dependencies needed once embeddings are cached).
3. **Aggregation** (`aggregate.py`) reuses the COLIPRI aggregator's pooled/per-fold-metric functions, extended to
   the four Rad-JEPA representations.

Run the full baseline (feature extraction once, then 10 folds x 4 representations, then aggregation) with:

```bash
CUDA_VISIBLE_DEVICES=3 bash bash/luna16_radjepa_3d/run_radjepa_baseline.sh
```

Results are written to:

- `outputs/luna16_radjepa_3d/embeddings/` — `per_patient/<seriesuid>.npz` caches (resize32 embedding, all sliding
  window embeddings + their start/end slice indices, native depth), `radjepa_embeddings_index.csv`,
  `radjepa_window_manifest.csv` (flat per-window index), and `extraction_config.json` (checkpoint path, HF config,
  embedding definition, environment info).
- `outputs/luna16_radjepa_3d/probes/fold_<N>/<representation>/` — per-fold probe checkpoints, training history,
  test metrics, and per-patient out-of-fold predictions.
- `outputs/luna16_radjepa_3d/probes/{pooled_metrics.csv,per_fold_mean_std_metrics.csv,radjepa_pooled_results.md}` —
  cross-fold pooled and mean/std reports.
- `results/09_radjepa_foundation_baseline/` — the curated copy of the final report.

## 3DINO Partial-Fine-Tuning Foundation-Model Baseline

`src/luna16_3dino_3d/` compares against a third 3D CT foundation model, [3DINO-ViT](https://github.com/AICONSlab/3DINO)
(Xu et al., npj Digital Medicine 2025, arXiv:2501.11755) -- a ViT-Large-3D (24 transformer blocks, embed_dim 1024)
pretrained with a DINOv2-style joint-embedding objective. Unlike COLIPRI/Rad-JEPA-3D (frozen encoder + linear
probe), this baseline uses **partial fine-tuning**: the first 12 blocks and the patch embedding are frozen, the
second 12 blocks + final norm + a new classification head are trained, with discriminative learning rates
(`--backbone-lr` ~1e-5 for the unfrozen pretrained blocks, `--head-lr` ~1e-4 for the new head). As with the other
two foundation-model baselines, no lesion-guided information (CPMNetv2 detections, nodule coordinates, masks) is
used -- only the full CT volume.

Every architectural/preprocessing detail below was verified directly against the official `AICONSlab/3DINO`
source and its `notebooks/basic_model_use.ipynb` demo (vendored, unmodified, under `external/3DINO/`, gitignored)
rather than assumed:

- **Native input size**: `112x112x112` (`dinov2/configs/train/vit3d_highres.yaml`, the high-resolution config the
  official demo notebook loads; `patch_size=16`, so 112/16=7 exactly, no padding needed for a single window).
- **Global representation**: the official demo calls the model directly, `out = model(x)` -> `(B, 1024)` -- i.e.
  the model's plain forward pass (`is_training=False`, default `head=Identity()`) already returns the post-norm
  CLS token. Used as-is for every window; no `get_intermediate_layers` multi-block sweep.
- **Preprocessing**: the exact per-window percentile normalization from the official notebook --
  `x = clip(((x - q0.05) / (q99.95 - q0.05)) * 2 - 1, -1, 1)` -- applied independently per window (matching how
  crops are normalized during SSL pretraining), on top of this repo's existing 1mm-isotropic preprocessed volumes
  (reusing `Luna16VolumeDataset` from `src/luna16_volume_3d/train_resnet18.py`, `volume_size=None` for the native
  shape).
- **Checkpoint loading**: `dinov2.utils.utils.load_pretrained_weights` (vendored, called directly, not
  reimplemented) -- `torch.load(path)["teacher"]`, strip `"module."`/`"backbone."` prefixes, `load_state_dict(strict=False)`.
- **Block-freezing** correctly accounts for the checkpoint's `block_chunks=4` module layout (`model.blocks` is 4
  `nn.ModuleList` chunks, each left-padded with `nn.Identity()` placeholders so a flat global block index is
  recoverable) -- verified against `vision_transformer.py` rather than assumed; see
  `model.py::flatten_transformer_blocks`.
- **xFormers**: not installed; the official code's `MemEffAttention` gracefully falls back to plain PyTorch
  attention when unavailable (`XFORMERS_DISABLED=1` set explicitly to avoid ambiguity) -- confirmed by reading
  `dinov2/layers/attention.py`, no separate/incompatible-torch environment needed (unlike Rad-JEPA-3D).

Pipeline:

1. **Windowing** (`model.py::extract_windows`): each patient's native CT volume is split into overlapping
   `112^3` windows with stride 56 (~50% overlap, configurable), including a flush final window per axis when a
   dimension isn't stride-divisible -- the same convention already used for Rad-JEPA-3D's sliding windows.
2. **Aggregation** (`lightning_model.py`): window CLS embeddings -> element-wise mean, element-wise max,
   concatenate(mean, max) -> `Linear(2048, 1)` head. No lesion-aware attention.
3. **Training** (`train.py`, PyTorch Lightning): one Lightning "batch" is one patient's full set of windows (never
   mixes patients), so `--accumulate-grad-batches` (default 8) reaches a reasonable effective batch size across
   patients. Supports mixed precision (`bf16-mixed` default), gradient clipping, `EarlyStopping`/`ModelCheckpoint`
   on `val_auc`, and a linear-warmup + cosine LR schedule. **Preprocessed windows are cached to disk**
   (`--windows-cache-dir`, shared across all 10 folds, since window content depends only on `(patient, window_size,
   stride)`) -- CPU-side volume loading/windowing/normalization was measured as the dominant per-patient cost
   (~3.5s) versus raw GPU compute (~0.5s for ~80 windows/patient on the A100), so caching removes that bottleneck
   after the first pass.
4. **Weights & Biases**: one run per fold via `WandbLogger`, named `3dino_partial_ft_fold{N}` (group
   `3dino_partial_ft`). Logs train/val/test loss, AUC, MCC, F1, accuracy, precision, recall, learning rate, epoch,
   trainable/frozen parameter counts, and GPU memory (`DeviceStatsMonitor`); the run config stores checkpoint
   path, frozen/trainable block counts, window size/stride, preprocessing, both learning rates, weight decay,
   batch size, gradient accumulation, seed, and fold ID. No patient-identifying information is logged.
5. **Aggregation** (`aggregate.py`) reuses the COLIPRI aggregator's pooled/per-fold-metric functions.

Run one fold with:

```bash
CUDA_VISIBLE_DEVICES=3 python src/luna16_3dino_3d/train.py --fold 0 \
  --checkpoint-path /path/to/3dino_vit_weights.pth --wandb
```

or all 10 folds with:

```bash
CHECKPOINT_PATH=/path/to/3dino_vit_weights.pth \
CUDA_VISIBLE_DEVICES=3 bash bash/luna16_3dino_3d/run_3dino_baseline.sh
```

**Checkpoint**: the official weights (`3dino_vit_weights.pth`, ~1.4GB) are gated on
[huggingface.co/AICONSlab/3DINO-ViT](https://huggingface.co/AICONSlab/3DINO-ViT) under CC BY-NC-ND 4.0 --
accept the license there with your own account and download the file (or use `huggingface_hub` with a token that
has accepted access); this repo does not vendor or auto-download it.

Results are written to:

- `outputs/luna16_3dino_3d/window_cache/` — cached preprocessed window tensors per patient (shared across folds).
- `outputs/luna16_3dino_3d/fold_<N>/3dino_partial_ft/` — config, freeze report, checkpoints, test metrics/predictions.
- `outputs/luna16_3dino_3d/{pooled_metrics.csv,per_fold_mean_std_metrics.csv,dino3d_pooled_results.md}` — cross-fold reports.
- `results/10_3dino_foundation_baseline/` — the curated copy of the final report.
