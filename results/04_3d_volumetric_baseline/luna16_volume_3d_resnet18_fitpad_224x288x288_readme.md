# LUNA16 Volumetric Baseline - 3D ResNet18

## Objective

This experiment provides a volumetric baseline for patient-level malignancy classification on LUNA16.

The goal is to compare:

```text
Full-volume 3D CNN
vs
Adaptive RBF 2D + EfficientNetV2-S
```

under the same patient-level labels, cross-validation splits, number of epochs, validation criterion, and pooled evaluation protocol.

## Data And Task

The baseline uses the same binary LUNA16 cohort used in the synthetic 2D experiments:

- benign: `476` scans
- malignant: `320` scans
- total binary cohort: `796` scans
- uncertain cases are excluded

The task is patient-level binary classification:

```text
benign vs malignant
```

The same 10-fold CSV split files are used:

```text
/ssd2/domenico/datasets/LUNA16_preprocessed/cv_splits/luna16_classification_fold*.csv
```

All preprocessed LUNA16 volumes have isotropic spacing:

```text
1.0 x 1.0 x 1.0 mm
```

This was verified from the NIfTI headers using SimpleITK.

## Input Standardization

The 3D baseline uses lung-cropped preprocessed CT volumes. Since the scans have variable spatial dimensions, a fixed input tensor is required for mini-batch training.

The selected target size is:

```text
D x H x W = 224 x 288 x 288
```

Volumes are standardized using:

```text
aspect-preserving isotropic scaling + centered zero padding
```

For each input volume with shape `(D, H, W)`, the scale factor is:

```text
scale = min(224 / D, 288 / H, 288 / W, 1.0)
```

The same scale factor is applied to all three axes. After resizing, the volume is centered and zero-padded to `224 x 288 x 288`.

This avoids:

- cropping out peripheral, apical, or basal nodules
- detector-derived or annotation-derived priors
- anisotropic deformation of the anatomy

Because the volumes are already resampled to `1 mm` isotropic spacing, preserving voxel-space aspect ratio also preserves physical anatomical proportions.

## Relation To The 2D Synthetic Resize

The 2D synthetic experiments use a different type of standardization. The synthetic images are 2D surface representations, not 3D anatomical volumes, and are resized before being passed to ImageNet-style 2D backbones.

The best synthetic 2D run uses:

```text
image_size = 256 x 384
```

This size was selected from the native dimensions of the generated 2D synthetic surfaces, not from the 3D fit-and-pad dimensions.

For the CPMNetv2 `top4, threshold=0.50` RBF synthetic surfaces, the native 2D grid dimensions were:

| Dimension | Min | 1% | 5% | 10% | 25% | 50% | 75% | 90% | 95% | 99% | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H | 167 | 176 | 183.4 | 189 | 201 | 216.5 | 233 | 247 | 254 | 267.3 | 282 |
| W | 220 | 247 | 257 | 265.7 | 277.8 | 299 | 322 | 337 | 345 | 365 | 395 |

Thus, `256 x 384` covers almost the entire native 2D surface range:

```text
H > 256: 31 / 888 cases
W > 384: 1 / 888 cases
```

The `224 x 288 x 288` size used by the volumetric baseline should therefore not be interpreted as the maximum size of the synthetic 2D images. It is a 3D tensor size chosen to standardize lung-cropped CT volumes while preserving nodule resolution. Conversely, `256 x 384` is a 2D image size chosen to match the natural resolution of the synthetic surface representations and to provide a fixed input size for 2D backbones.

## Why Not Center Crop?

A center crop is not appropriate for this baseline because nodules can be peripheral or located near the lung apex/base. A fixed center crop could remove clinically relevant regions.

The adopted fit-and-pad strategy keeps the whole lung-cropped volume inside the target tensor. Therefore, nodules are not removed by the preprocessing.

## Why 224 x 288 x 288?

The target size was selected as a compromise between anatomical preservation and computational feasibility on the A100 GPU.

Smaller target sizes, such as `160 x 224 x 224`, are computationally cheaper but downscale some small nodules too aggressively. In particular, with `160 x 224 x 224`, `23.4%` of nodule mask components would have a post-scaling minimum dimension below `3` voxels.

Larger target sizes would preserve even more spatial detail, but the cost of a 3D CNN grows rapidly with the number of voxels. Moving from `224 x 288 x 288` to a larger volume such as `256 x 320 x 320` would substantially increase activation memory and training time across all convolutional layers. This is especially important because the baseline is trained over 10 folds for 100 epochs, with an effective batch size of 8.

Using the native lung-cropped volumes without standardization was also tested conceptually. It would avoid any downscaling, but it forces `batch_size = 1` because volumes have different shapes and makes training prohibitively slow for a full 10-fold experiment. The native-volume version was therefore not a practical baseline for exhaustive cross-validation.

The selected target `224 x 288 x 288` is the largest tested configuration that remained practical on the A100 while preserving most nodule resolution:

```text
batch_size = 2
accumulate_grad_batches = 4
effective_batch_size = 8
mixed precision = 16-mixed
GPU memory usage ~= 24 GB on A100
```

This choice keeps the experiment feasible while avoiding the stronger nodule-resolution degradation observed with smaller targets.

With `224 x 288 x 288`, the scale-factor distribution over the `796` binary scans is:

| Percentile | Scale |
|---:|---:|
| min | 0.614 |
| 1% | 0.644 |
| 5% | 0.687 |
| 10% | 0.707 |
| 25% | 0.746 |
| 50% | 0.789 |
| 75% | 0.842 |
| 90% | 0.892 |
| 95% | 0.949 |
| 99% | 1.000 |
| max | 1.000 |

No scan has a scale factor below `0.60`, and only `8.2%` of scans have a scale factor below `0.70`.

## Resulting Size Before Padding

After aspect-preserving scaling and before zero padding, the output dimensions are typically:

```text
D x H x W ~= 224 x 171 x 236
```

Dimension percentiles before padding:

| Dimension | Min | 5% | 50% | 95% | Max |
|---|---:|---:|---:|---:|---:|
| D | 118 | 224 | 224 | 224 | 224 |
| H | 125 | 141 | 171 | 210 | 248 |
| W | 174 | 202 | 236 | 283 | 288 |

Examples near the median scale factor:

| Original D,H,W | After Scaling D,H,W | Padding To Target |
|---:|---:|---:|
| 284, 208, 274 | 224, 164, 216 | 0, 124, 72 |
| 284, 228, 320 | 224, 180, 252 | 0, 108, 36 |
| 284, 212, 311 | 224, 167, 245 | 0, 121, 43 |
| 284, 200, 267 | 224, 158, 211 | 0, 130, 77 |

Worst scaling examples:

| Original D,H,W | Scale | After Scaling D,H,W | Padding To Target |
|---:|---:|---:|---:|
| 365, 234, 310 | 0.614 | 224, 144, 190 | 0, 144, 98 |
| 362, 240, 305 | 0.619 | 224, 149, 189 | 0, 139, 99 |
| 360, 226, 337 | 0.622 | 224, 141, 210 | 0, 147, 78 |
| 354, 244, 341 | 0.633 | 224, 154, 216 | 0, 134, 72 |

Even in the worst cases, the in-plane dimensions before padding remain approximately `H=141-154` and `W=189-216`.

## Nodule Information Preservation

To quantify potential loss of nodule resolution, nodule mask components were analyzed after applying the same scale factor used for the CT volume.

The analysis considered:

```text
701 nodule mask components
```

Post-scaling minimum nodule dimension percentiles:

| Percentile | Minimum Dimension, voxels |
|---:|---:|
| min | 2.12 |
| 1% | 2.55 |
| 5% | 3.10 |
| 10% | 3.39 |
| 25% | 4.26 |
| 50% | 6.20 |
| 75% | 9.86 |
| 90% | 14.75 |
| 95% | 16.82 |
| 99% | 20.68 |
| max | 27.27 |

Counts below resolution thresholds:

| Threshold | Components | Fraction |
|---:|---:|---:|
| `< 2 voxels` | 0 / 701 | 0.0% |
| `< 3 voxels` | 25 / 701 | 3.6% |
| `< 4 voxels` | 143 / 701 | 20.4% |
| `< 5 voxels` | 258 / 701 | 36.8% |

Thus, no annotated nodule component collapses below `2` voxels, and only `3.6%` of components fall below `3` voxels in minimum dimension.

Compared with the smaller `160 x 224 x 224` target, where `23.4%` of nodule components fell below `3` voxels, the selected `224 x 288 x 288` target substantially reduces nodule-resolution degradation.

## Training Configuration

The volumetric baseline is trained as:

```text
model: 3D ResNet18
input: 224 x 288 x 288
batch_size: 2
accumulate_grad_batches: 4
effective_batch_size: 8
epochs: 100
learning_rate: 1e-4
weight_decay: 1e-4
precision: 16-mixed
monitor: val_mcc
checkpoint: best validation MCC
```

The current run is:

```text
outputs/luna16_volume_3d_resnet18_fitpad_224x288x288_b2_acc4_ep100_wandb
```

W&B group:

```text
resnet18_fitpad_224x288x288_b2_acc4_ep100_wandb
```

## Scientific Justification

This preprocessing makes the volumetric baseline scientifically defensible because it satisfies four constraints:

1. It preserves the full lung-cropped field of view.
2. It does not use detector outputs, candidate locations, or nodule annotations.
3. It avoids cropping out peripheral nodules.
4. It avoids anisotropic anatomical deformation by applying a single scale factor across all axes.

A concise description for the paper is:

> For the volumetric baseline, all LUNA16 CT volumes were first resampled to isotropic 1 mm spacing and lung-cropped during preprocessing. To enable mini-batch training, each volume was standardized to a fixed tensor size using isotropic aspect-preserving scaling followed by centered zero padding. The target size was set to 224 x 288 x 288, which preserved the full lung-cropped field of view while minimizing nodule-resolution loss: no annotated nodule component was reduced below 2 voxels in minimum dimension, and only 3.6% were reduced below 3 voxels. This baseline did not use detector outputs or nodule annotations and was evaluated with the same patient-level labels, cross-validation splits, validation criterion, and pooled metrics as the proposed 2D synthetic representation.
