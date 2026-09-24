# Experiment launchers

All shell launchers live here, one folder per `src/` package. Every script
sources [`common.sh`](common.sh), so it can be run from any directory:

```bash
bash bash/<folder>/<script>.sh [extra CLI arguments]
```

`common.sh` moves to the repository root, activates the first virtual
environment it finds (`$VENV_PATH`, `myenv/`, `.venv/`, `venv/`; set
`SKIP_VENV=1` to skip), adds the root to `PYTHONPATH`, and defines the default
data layout. Configuration is passed via environment variables; every default
can be overridden:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `data/processed` | Preprocessed LUNA16 volumes and masks |
| `SPLITS_DIR` | `$DATA_ROOT/cv_splits` | Patient-level 10-fold split CSVs |
| `SYNTHETIC_ROOT` | `data/synthetic_2d` | Generated synthetic 2D images |
| `DETECTOR_PREDICTIONS` | `outputs/cpmnetv2_luna16_10fold_bf16_guarded_results/normalized_predictions` | Out-of-fold CPMNetv2 candidates |
| `FOLDS` | `0 1 2 3 4 5 6 7 8 9` | Folds to run |
| `DEVICES` / `CUDA_VISIBLE_DEVICES` | `[0]` / `0` | GPU selection |

Example: `DATA_ROOT=/datasets/LUNA16_preprocessed FOLDS="0 1" DEVICES="[1]" bash bash/luna16_synthetic_2d/train_top4_minprob0.5_rbf.sh`

## Pipeline order

The proposed method (adaptive RBF, top-4, p >= 0.5, EfficientNetV2-S):

```bash
bash bash/preprocessing/luna16_preprocessing.sh           # 1. resample, crop to lungs, write NIfTI + labels
bash bash/preprocessing/reorganize_luna16_subsets.sh      # 2. group scans by LUNA16 subset
bash bash/cpmnetv2/train_10fold.sh                        # 3. train the detector on every fold
bash bash/luna16_synthetic_2d/generate_top4_minprob0.5_rbf.sh  # 4. build the synthetic 2D images
bash bash/luna16_synthetic_2d/train_top4_minprob0.5_rbf.sh     # 5. train the 2D classifiers
bash bash/luna16_synthetic_2d/gradcam_top4_minprob0.5_rbf.sh   # 6. Grad-CAM explanations
```

Step 4 reads the out-of-fold detector predictions of step 3 from
`DETECTOR_PREDICTIONS` (one `*fold<k>/predictions/test_predictions.csv` per fold).
When a scan has no detector candidate, step 4 places pseudo nodules sampled from
an empirical distribution
(`outputs/luna16_saliency_control_point_distribution/empirical_nodule_distribution_from_control_points.npz`),
built beforehand from the control points of the ground-truth-guided synthetic
images:

```bash
bash bash/luna16_synthetic_2d/generate_gt_synthetic_images.sh       # -> outputs/luna16_saliency_synthetic_gt
bash bash/luna16_synthetic_2d/build_nodule_position_distribution.sh # -> the .npz above
```

If the `.npz` does not exist yet, the generators print a warning and fall back
to pseudo nodules without the empirical prior, so the GT images can be built first.

## Scripts

| Folder | Script | Paper experiment |
|---|---|---|
| `preprocessing/` | `luna16_preprocessing.sh` | Isotropic resampling and lung cropping |
| | `reorganize_luna16_subsets.sh` | Subset folder layout |
| `cpmnetv2/` | `train_fold.sh` | Detector on one split (used by `train_10fold.sh`) |
| | `train_10fold.sh` | CPMNetv2 detector, 10 folds |
| `luna16_synthetic_2d/` | `generate_synthetic_images.sh` | Generic generator (`SURFACE_METHOD`, `CONTROL_POINT_MODE`, `TOP_K`, `MIN_PROBABILITY`) |
| | `generate_gt_synthetic_images.sh` | Ground-truth-guided images (input of the nodule distribution) |
| | `build_nodule_position_distribution.sh` | Empirical nodule distribution for pseudo nodules in detector-negative scans |
| | `generate_top4_minprob0.5_{rbf,shepard}.sh` | Adaptive RBF / Shepard images |
| | `generate_top4_minprob0.5_rbf_{fixed,random}_control.sh` | Fixed- / random-control RBF ablation images |
| | `generate_top4_minprob0.5_rbf_control_ablation.sh` | Both ablations, sequentially |
| | `train_backbones.sh` | Generic 2D trainer (`EXPERIMENT_NAME`, `BACKBONES`) |
| | `train_top4_minprob0.5_{rbf,shepard}.sh` | Adaptive RBF / Shepard classifiers |
| | `train_top4_minprob0.5_rbf_{fixed,random}_control.sh` | Fixed- / random-control RBF classifiers |
| | `train_top4_minprob0.5_rbf_control_ablation.sh` | Both ablations in parallel (`FIXED_DEVICES`, `RANDOM_DEVICES`) |
| | `gradcam_top4_minprob0.5_rbf.sh` | Grad-CAM of the proposed method |
| `luna16_2d_baselines/` | `run_mip_efficientnetv2s.sh` | MIP axial / tri-view and central slice (`MODES`) |
| `luna16_slice_attention_2p5d/` | `run_slice_attention_effnetv2s.sh` | Soft slice attention |
| `luna16_detection_mil/` | `run_detection_mil_efficientnetv2s.sh` | Detector crop-MIL |
| | `run_detection_mil_effnetv2s_preprocessed_uint8.sh` | Detector crop-MIL, uint8 crops |
| `luna16_volume_3d/` | `run_resnet18_10fold.sh` | 3D ResNet18 |
| `luna16_radjepa_3d/` | `run_radjepa_baseline.sh` | Rad-JEPA-3D linear probes (needs `RADJEPA_PYTHON`) |
| `luna16_3dino_3d/` | `run_3dino_baseline.sh` | 3DINO-ViT partial fine-tuning (needs `CHECKPOINT_PATH`) |
