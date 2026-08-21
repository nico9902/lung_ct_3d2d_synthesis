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

The current paper draft is in [`latex/main.tex`](latex/main.tex), with the compiled PDF in [`latex/main.pdf`](latex/main.pdf).

## Repository Layout

```text
bash/                         Experiment launch scripts
docs/                         Method notes, QC reports, reproducibility notes
latex/                        Paper draft and figures
results/                      Curated result summaries for paper writing
src/cls/                      Generic 2D classification components
src/det/                      Detector-related code and integrations
src/luna16_synthetic_2d/      Adaptive 2D synthesis and 2D backbone experiments
src/luna16_detection_mil/     Detector-crop MIL baseline
src/luna16_volume_3d/         3D ResNet18 volumetric baseline
src/prs/                      Preprocessing and label-generation utilities
```

Large local folders such as `data/`, `outputs/`, `wandb/`, `wandb2/`, model checkpoints, medical images, and NumPy arrays are intentionally ignored by Git.

## Main Results

Curated paper-ready summaries are organized under [`results/`](results/):

- [`results/00_paper_overview`](results/00_paper_overview): master result tables.
- [`results/01_synthetic_2d_adaptive`](results/01_synthetic_2d_adaptive): adaptive RBF/Shepard and control-point ablations.
- [`results/02_detector_and_working_points`](results/02_detector_and_working_points): detector FROC and working-point analysis.
- [`results/03_2d_nonadaptive_baselines`](results/03_2d_nonadaptive_baselines): MIP, central slice, and crop-MIL baselines.
- [`results/04_3d_volumetric_baseline`](results/04_3d_volumetric_baseline): 3D ResNet18 baseline.
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
pip install -r requirements_saliency.txt
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

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the expected artifact organization and the main experiment commands.

## Paper Draft

The LaTeX manuscript is kept in [`latex/`](latex/). To compile:

```bash
cd latex
latexmk -pdf -interaction=nonstopmode main.tex
```

The current method figure is [`latex/figures/Method.pdf`](latex/figures/Method.pdf).

## Notes For GitHub Use

Before committing, check:

```bash
git status --short
```

Do not commit patient data, checkpoints, W&B runs, generated synthetic datasets, or large binary arrays. The repository is intended to track code, paper sources, scripts, and curated result summaries only.

