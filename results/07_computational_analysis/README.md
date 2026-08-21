# Computational Analysis

This folder summarizes the computational profile of the main LUNA16 experiments. The goal is to support the methodological claim that the adaptive synthetic 2D representation is not only more accurate, but also computationally practical compared with volumetric, projection-based, and detector-crop alternatives.

## Files

- `computational_profile.csv`: model/data/runtime profile for each main method.
- `input_complexity.csv`: input tensor size and forward-pass burden per patient.
- `flops_profile.csv`: MACs and GFLOPs for the main classifier-side forward passes.
- `detector_sliding_window_profile.csv`: CPMNetv2 sliding-window crop counts and detector-inclusive GFLOPs.
- `volume_shape_summary.csv`: shape statistics of the preprocessed LUNA16 volumes.

## Key Observations

The proposed Adaptive RBF representation reduces each CT volume to a single `3 x 256 x 384` image after detector inference. Therefore, two computational views should be reported:

- **Classifier-side cost**: cost after detector predictions and synthetic images are available.
- **Detector-inclusive cost**: cost starting from the CT volume, including CPMNetv2 sliding-window inference.

The classifier-side input is the same size as MIP and central-slice baselines, but with a much more informative detector-guided geometry.

| Method | Input burden per patient | Relative to proposed 2D | AUC |
|---|---:|---:|---:|
| Adaptive RBF | 1 image | 1.0x | **0.8149** |
| MIP axial / central slice | 1 image | 1.0x | 0.6075 / 0.6153 |
| Detection crop-MIL top4 | up to 4 crops | 4.0x | 0.5410 |
| 3D ResNet18 fit-pad | 1 volume, 224x288x288 | 63.0x scalar input | 0.5659 |

The 3D baseline uses a tensor of `1 x 224 x 288 x 288`, corresponding to `18,579,456` scalar input values per patient. The proposed 2D image contains `294,912` scalar values, so the 3D input is approximately `63x` larger before considering 3D convolutional activations.

## FLOPs

FLOPs are reported using the explicit convention `GFLOPs = 2 x MACs`, where one multiply-add contributes two floating-point operations. The corresponding MAC count is also saved in `flops_profile.csv`.

| Method | Backbone | Input | GMACs | GFLOPs |
|---|---|---|---:|---:|
| Adaptive RBF / Shepard, classifier only | EfficientNetV2-S | `1 x 3 x 256 x 384` | 5.5774 | 11.1549 |
| CPMNetv2 detector, per crop | CPMNetv2 | `1 x 1 x 64 x 128 x 128` | 343.7484 | 687.4968 |
| CPMNetv2 detector, mean per scan | CPMNetv2 | mean `48.40` crops/scan | 16635.7957 | 33271.5914 |
| Adaptive RBF, detector-inclusive | CPMNetv2 + EfficientNetV2-S | mean detector scan + 2D classifier | 16641.3731 | 33282.7463 |
| MIP / central slice | EfficientNetV2-S | `1 x 3 x 256 x 384` | 5.5774 | 11.1549 |
| Detection crop-MIL top4 | EfficientNetV2-S | `4 x (1 x 3 x 256 x 384)` | 22.3098 | 44.6196 |
| 3D ResNet18 fit-pad | R3D-18 | `1 x 1 x 224 x 288 x 288` | 3679.9214 | 7359.8427 |

Thus, if considering only the final patient classifier, the 3D ResNet18 forward pass is approximately `659.8x` more expensive in GFLOPs than the proposed single-image Adaptive RBF classifier pass. However, if the detector is included, the adaptive pipeline is dominated by CPMNetv2 sliding-window inference: the detector-inclusive adaptive pipeline is approximately `4.52x` more expensive than the 3D ResNet18 forward pass in neural-network GFLOPs (`33282.75` vs `7359.84` GFLOPs).

This distinction is important. The adaptive method is computationally light at the classification stage and storage-efficient after synthetic generation, but it requires a detector pass when starting from raw/preprocessed CT volumes. In this study, detector predictions are also a reusable intermediate: the same CPMNetv2 outputs support FROC analysis, working-point selection, nodule coverage analysis, synthetic generation, and detector-crop MIL baselines.

## Detector Sliding-Window Profile

CPMNetv2 uses crop size `64 x 128 x 128` with overlap `16 x 32 x 32`, corresponding to stride `48 x 96 x 96`. Across the `888` preprocessed LUNA16 scans:

| Quantity | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Detector windows per scan | 48.40 | 38 | 18 | 96 |
| Detector GFLOPs per scan | 33271.59 | 26124.88 | 12374.94 | 65999.69 |

## Storage

Measured on `/ssd2`:

| Dataset representation | Storage |
|---|---:|
| LUNA16 preprocessed volumes | 54G |
| Adaptive RBF synthetic 2D | 394M |
| Adaptive Shepard synthetic 2D | 395M |
| MIP axial 2D baseline | 63M |
| MIP tri-view 2D baseline | 104M |
| Central axial slice baseline | 55M |

The adaptive synthetic representation is roughly two orders of magnitude smaller than the preprocessed volumetric dataset while preserving substantially more discriminative information than fixed 2D projections.

## Runtime Evidence

The 3D ResNet18 fit-pad baseline logged `epoch_seconds` for every fold. Across `1000` fold-epochs:

| Statistic | Seconds/epoch |
|---|---:|
| Mean | 254.94 |
| Median | 298.99 |
| Min | 182.47 |
| Max | 340.65 |

This corresponds to roughly `4.25` minutes per epoch on average, using the A100 configuration `batch_size=4`, `accumulate_grad_batches=2`, `precision=16-mixed`, and volume size `224x288x288`.

For the 2D classifiers, the available Lightning `metrics.csv` files do not include wall-clock timestamps. Therefore, this report does not claim exact 2D seconds per epoch. The reliable comparison is instead based on input size, storage, number of encoder forward passes per patient, and measured 3D runtime.

## Paper Interpretation

The computational analysis supports four points:

1. The proposed method has the same classifier-side input cost as ordinary 2D baselines, but much higher accuracy.
2. At the classifier stage, it is far cheaper than the full-volume 3D baseline in GFLOPs.
3. End-to-end from CT volume, the detector must be included and dominates the adaptive pipeline cost.
4. The detector cost is a reusable/precomputable intermediate rather than a cost unique to classification alone.

A concise paper statement could be:

> The adaptive synthetic representation compresses each 1 mm isotropic lung CT volume into a single 256 x 384 2D image, reducing the final classifier input dimensionality by approximately 63-fold relative to the 224 x 288 x 288 volumetric baseline. At the classifier level, the proposed EfficientNetV2-S forward pass requires 11.15 GFLOPs, whereas the 3D ResNet18 volumetric baseline requires 7359.84 GFLOPs. When detector inference is included, the adaptive pipeline requires approximately 33282.75 GFLOPs per scan on average, dominated by CPMNetv2 sliding-window inference. This detector cost is reusable across detection evaluation, working-point selection, nodule coverage analysis, and synthetic image generation. Despite the detector-inclusive cost, the adaptive RBF representation significantly outperformed 3D ResNet18, MIP, central-slice, detector-crop MIL, and unguided RBF ablations.
