# Reproducibility Notes

This document summarizes how the repository is organized for reproducing the main experiments. It intentionally separates code tracked by Git from local data and generated artifacts that are too large or sensitive for the repository.

## Tracked By Git

The repository should track:

- source code under `src/`;
- experiment launch scripts under `bash/`;
- paper sources under `latex/`;
- curated markdown/CSV summaries under `results/`;
- documentation under `docs/`;
- small configuration files.

## Not Tracked By Git

The following should remain local:

- raw LIDC-IDRI/LUNA16 CT data;
- preprocessed CT volumes;
- generated synthetic image datasets;
- detector predictions and full training outputs;
- W&B logs;
- model checkpoints;
- NumPy arrays and medical-image files.

The expected local folders are:

```text
data/LUNA16/
data/LUNA16_preprocessed/
data/LIDC-IDRI files/
outputs/
wandb/
wandb2/
```

## Main Experiment Families

### Adaptive 2D Synthesis

Code:

```text
src/luna16_synthetic_2d/
```

Launch scripts:

```text
bash/run_luna16_detector_saliency_top4_minprob0.5_rbf.sh
bash/run_luna16_detector_saliency_top4_minprob0.5_shepard.sh
bash/run_luna16_detector_saliency_top4_minprob0.5_rbf_fixed.sh
bash/run_luna16_detector_saliency_top4_minprob0.5_rbf_random.sh
src/luna16_synthetic_2d/run_backbones_det_top4_minprob0.5_rbf.sh
src/luna16_synthetic_2d/run_backbones_det_top4_minprob0.5_shepard.sh
```

The main paper configuration is:

```text
detector: CPMNetv2
threshold: 0.50
top-k: 4
surface: RBF
classifier: EfficientNetV2-S
```

### Detector Crop-MIL Baseline

Code:

```text
src/luna16_detection_mil/
```

This baseline uses detector-centered crops, a shared 2D encoder, and patient-level pooling.

### Non-Adaptive 2D Baselines

Code:

```text
src/luna16_synthetic_2d/generate_mip_baselines.py
```

Baselines include axial MIP, tri-view MIP, and central axial slice.

### 3D Volumetric Baseline

Code:

```text
src/luna16_volume_3d/
```

Launch scripts:

```text
bash/luna16_volume_3d/
```

The current volumetric baseline is a 3D ResNet18 trained on fit-padded `224 x 288 x 288` volumes.

## Curated Results

Paper-ready outputs are under:

```text
results/
```

Important files:

```text
results/00_paper_overview/master_pooled_results.csv
results/01_synthetic_2d_adaptive/luna16_synthetic_2d_cpmnetv2_bf16_top4_minprob0.50_pooled_results.md
results/02_detector_and_working_points/README_results.md
results/06_statistical_tests/README.md
results/07_computational_analysis/README.md
```

## Paper Compilation

Compile from the `latex/` folder:

```bash
cd latex
latexmk -pdf -interaction=nonstopmode main.tex
```

Only LaTeX sources, figures, and the final paper PDF should be committed. Build intermediates such as `.aux`, `.fls`, `.fdb_latexmk`, `.out`, `.log`, and `.spl` are ignored.

## Before Publishing

Run:

```bash
git status --short
```

Check that no private paths, credentials, raw patient data, generated synthetic datasets, checkpoints, or W&B runs are staged.

