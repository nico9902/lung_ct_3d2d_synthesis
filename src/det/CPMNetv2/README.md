# CPMNetv2 3D Nodule Detector

This directory contains a CPMNetv2-style 3D detector adapted for LIDC-IDRI lung
nodule detection and wrapped with PyTorch Lightning/Hydra. The implementation is
an anchor-free, center-point detector: each voxel of the output feature grid is a
candidate object center, and the network predicts an objectness logit, a 3D box
size, and a sub-grid center offset for that candidate.

The most relevant files are:

| File | Role |
| --- | --- |
| `networks/ResNet_3D_CPM.py` | 3D ResNet/FPN-like detector, detection loss, anchor/center generation, and postprocessing. |
| `lightning_model.py` | Lightning module: loss weighting, optimizer/scheduler, inference, prediction CSV writing, FROC invocation. |
| `lidc_datamodule.py` | LIDC-IDRI split handling and Lightning dataloaders. |
| `dataload/dataset_lidc.py` | LIDC NIfTI loading, view conversion, label conversion, crops, and split-combine inference dataset. |
| `evaluationScript/detectionCADEvalutionIOU.py` | IOU-based CAD/FROC evaluation. |
| `conf/train_lightning.yaml` | Default Hydra configuration. |
| `bash/cpmnetv2_train_lightning_hydra.sh` | Example shell launcher with practical training defaults. |

## Model Overview

CPMNetv2 is a dense 3D object detector for small volumetric lesions. Instead of
placing many hand-designed anchors of different sizes at each location, it
represents every object by:

- a center point `(z, y, x)`;
- a 3D box size `(d, h, w)`;
- a local offset from the feature-grid point to the real center;
- an objectness probability.

For a crop of size `[D, H, W]`, the network produces a lower-resolution feature
map with shape `[D', H', W']`. Every grid location is converted to an anchor
point by `make_anchors(...)`. The stride is inferred from the crop size and
feature-map size:

```text
stride = [D / D', H / H', W / W']
anchor point = [grid_z, grid_y, grid_x]
predicted center = (anchor point + predicted offset) * stride
predicted size = 2 * predicted shape
predicted box = [center_z, center_y, center_x, d, h, w]
```

The factor `2 * predicted shape` exists because the regression target stores
half-sizes during assignment.

## Network Architecture

The default detector is `resnet18` in `networks/ResNet_3D_CPM.py`.

Input tensors have shape:

```text
[batch, channels=1, depth, height, width]
```

For the LIDC Lightning path, axial volumes are loaded as `(z, y, x)`, clipped to
HU `[-200, 800]`, normalized to `[0, 1]`, and then mapped to `[-1, 1]`.

The backbone is a compact 3D encoder-decoder:

1. `in_conv`: 3D convolution stem.
2. `in_dw`: first downsampling convolution. In the Lightning wrapper this uses
   `first_stride=(1, 2, 2)`, preserving depth resolution early while reducing
   in-plane resolution.
3. Encoder stages:
   - `block1`, `block1_dw`
   - `block2`, `block2_dw`
   - `block3`, `block3_dw`
   - `block4`
4. Decoder stages:
   - `block33_up`, skip connection with processed `block3` features;
   - `block22_up`, skip connection with processed `block2` features.
5. Detection head `ClsRegHead`.

The residual blocks are `BasicBlockNew` modules with two 3D convolutions,
normalization, activation, optional squeeze-and-excitation (`se=True`), and a
residual path. Normalization and activation are configurable:

```yaml
norm_type: batchnorm
head_norm: batchnorm
act_type: ReLU
se: false
```

The detection head has three independent branches:

| Branch | Output channels | Meaning |
| --- | ---: | --- |
| `Cls` | 1 | Objectness logit per grid point. |
| `Shape` | 3 | Half-size-like box shape prediction in `(d, h, w)` order before doubling. |
| `Offset` | 3 | Offset from grid point to object center in feature-grid units. |

The head initialization encodes useful priors:

- classification bias is initialized with prior probability `0.01`;
- shape bias starts at `0.5`;
- offset bias starts at `0.05`.

These priors reduce early training instability because the dense grid is mostly
background.

## Data And Coordinates

The LIDC adapter expects a split CSV:

```text
patient_id,nodule_count,target,split
LIDC-IDRI-0001,1,True,train
LIDC-IDRI-0002,0,False,val
...
```

and case folders:

```text
LIDC-IDRI_nifti/
  LIDC-IDRI-0001/
    LIDC-IDRI-0001_volume.nii.gz
    LIDC-IDRI-0001_nodule_mask.nii.gz
```

Training labels come from `lidc_labels.csv`:

```text
seriesuid,x,y,z,w,h,d,label
LIDC-IDRI-0001,98.5,211.0,54.0,8.0,8.0,6.0,nodule
```

CSV labels are stored in world/image axis order `(x, y, z, w, h, d)`. The model
internally uses tensor order `(z, y, x, d, h, w)`. `dataset_lidc.py` maps label
columns according to the requested view:

| View | Tensor location columns | Tensor size columns |
| --- | --- | --- |
| `axial` | `z, y, x` | `d, h, w` |
| `coronal` | `y, x, z` | `h, w, d` |
| `sagittal` | `x, y, z` | `w, h, d` |

The default view is `axial`, which is CPMNet-native.

### Training Crops

Training uses `InstanceCrop` followed by augmentations:

- random flip along depth/height/width;
- random transpose in the axial plane;
- pad to the requested crop size;
- random crop with positive sampling bias;
- conversion from coordinates to padded annotation tensors.

Important crop parameters:

```yaml
crop_size: [64, 128, 128]
spacing: [0.7, 0.3125, 0.3125]
num_samples: 1
topk: 5
```

In the shell launcher, a more memory-intensive setting is used:

```bash
BATCH_SIZE="2"
NUM_SAMPLES="10"
ACCUMULATE_GRAD_BATCHES="8"
TOPK="7"
SPACING="[1.0, 1.0, 1.0]"
```

`num_samples` is the number of random crops generated per case. The effective
number of crop tensors per optimization step is approximately:

```text
batch_size * num_samples * accumulate_grad_batches
```

## Detection Loss

The loss is implemented by `Detection_loss` in `networks/ResNet_3D_CPM.py`.
The Lightning wrapper computes the weighted sum:

```text
L = lambda_cls    * L_cls
  + lambda_shape  * L_shape
  + lambda_offset * L_offset
  + lambda_iou    * L_iou
```

Default YAML weights:

```yaml
lambda_cls: 4.0
lambda_shape: 0.1
lambda_offset: 1.0
lambda_iou: 1.0
```

The provided shell launcher uses:

```bash
LAMBDA_CLS="1.0"
LAMBDA_SHAPE="1.0"
LAMBDA_OFFSET="1.0"
LAMBDA_IOU="2.0"
```

### 1. Annotation Preprocessing

`target_proprocess(...)` clips each ground-truth box to the current crop. Boxes
are represented as:

```text
[center_z, center_y, center_x, d, h, w, class]
```

A clipped box is kept only when:

```text
visible_volume / original_volume > 0.1
clipped_volume >= 15 voxels
```

If a box is partially present but does not satisfy these constraints, the
corresponding crop region is marked as ignored. Ignored voxels do not contribute
to the classification loss. This avoids punishing the model for ambiguous border
fragments.

### 2. Center-Point Assignment

`get_pos_target(...)` assigns each ground-truth object to the nearest feature
grid points. Distance is computed in physical spacing-aware units:

```text
distance = -sum(((gt_center / stride - anchor_point) * spacing) ** 2)
```

For each object:

- the nearest `topk` grid points are positives;
- the next `ignore_ratio * topk` points are ignored;
- all other valid points are negatives.

With default `ignore_ratio=5` and `topk=5`, each object contributes up to `5`
positive center points and `25` ignored neighbor points.

The assignment produces:

| Target | Meaning |
| --- | --- |
| `target_scores` | 1 for positive grid points, 0 for negatives. |
| `target_offset` | `gt_center / stride - anchor_point`. |
| `target_shape` | half-size target `(d, h, w) / 2`. |
| `target_bboxes` | full ground-truth box in crop coordinates. |
| `mask_ignore` | grid points excluded from classification. |

### 3. Classification Loss

`cls_loss(...)` is a focal-style binary classification loss over dense grid
points:

```text
BCEWithLogits(pred_score, target_score)
focal_weight = alpha_t * (1 - p_t) ** gamma
```

with:

```text
alpha = 0.75
gamma = 2.0
```

The implementation adds two hard-example mechanisms:

1. Positive points predicted with probability `< 0.8` are upweighted by
   `FN_weights = 4.0`.
2. Negatives are subsampled and hard-mined:
   - at most `num_neg = 10000` negatives are sampled;
   - when positives exist, the top `ratio * num_positive_pixels` negative losses
     are kept, with `ratio = 100`;
   - when no positives exist, the top `num_hard = 100` negative losses are kept.

The final classification loss for each batch item is normalized by:

```text
max(number_of_positive_points, 1)
```

This is important for nodule detection because most crops contain far more
background centers than lesion centers.

### 4. Shape Loss

Shape regression is an L1 loss on positive grid points only:

```text
L_shape = mean(abs(pred_shape - target_shape))
```

`target_shape` is half of the ground-truth box size in crop coordinates.

### 5. Offset Loss

Offset regression is also an L1 loss on positive grid points only:

```text
L_offset = mean(abs(pred_offset - target_offset))
```

This teaches the detector to move from the coarse feature-grid point to the
continuous object center.

### 6. DIoU Loss

Predicted boxes are decoded as:

```text
center = (anchor_point + pred_offset) * stride
size = 2 * pred_shape
box = [center_z, center_y, center_x, d, h, w]
```

The IOU term uses distance-IoU (`DIoU=True`):

```text
DIoU = IoU - center_distance_squared / enclosing_box_diagonal_squared
L_iou = -mean(DIoU)
```

Because the loss is negative DIoU, better overlap and better center alignment
make the term smaller.

If a crop has no foreground points, all three regression losses are set to zero
and only the classification loss is optimized.

## Inference And Postprocessing

Inference uses `Detection_Postprocess`.

For each split crop:

1. Apply sigmoid to `Cls`.
2. Decode boxes from `Shape` and `Offset`.
3. Keep the top `post_topk` scores.
4. Remove predictions below `post_threshold`.
5. Apply 3D NMS with `post_nms_threshold`.
6. Keep at most `post_num_topk` detections per split.

Lightning defaults:

```text
post_topk = 60
post_threshold = 0.15
post_nms_threshold = 0.05
post_num_topk = 20
```

The evaluation dataset runs full CT inference by splitting each scan into
overlapping crops:

```yaml
crop_size: [64, 128, 128]
overlap_size: [16, 32, 32]
```

After every crop is processed, `SplitComb.combine(...)` maps crop-local
detections back to the full-volume coordinate system. A final scan-level NMS is
then applied:

```text
final_nms_overlap = 0.05
final_topk = 40
```

Prediction CSVs are written as:

```text
seriesuid,coordX,coordY,coordZ,probability,w,h,d
```

under:

```text
outputs/cpmnetv2/<experiment_name>/predictions/
```

## FROC Computation

FROC is computed by `noduleCADEvaluation(...)` in
`evaluationScript/detectionCADEvalutionIOU.py`. The Lightning module calls it at
the end of testing when `no_froc: false`.

The evaluator consumes:

| Input | Columns / content |
| --- | --- |
| annotation CSV | `seriesuid,coordX,coordY,coordZ,w,h,d` |
| excluded annotation CSV | same columns; detections on these boxes can be ignored |
| seriesuid CSV | one scan ID per row |
| prediction CSV | `seriesuid,coordX,coordY,coordZ,probability,w,h,d` |

For the LIDC adapter, ground-truth files are generated automatically from the
test dataset labels into:

```text
predictions/test_gt/
```

### Candidate Matching

For each scan, predicted candidates are matched to included ground-truth nodules
using 3D bounding-box IOU. The current default threshold is:

```yaml
froc_iou_threshold: 0.1
```

A prediction is a true positive if:

```text
IoU(predicted_box, ground_truth_box) >= froc_iou_threshold
```

If multiple candidates match the same nodule, the evaluator uses the highest
probability candidate for FROC and counts the remaining matches as duplicate
detections. Candidates matching excluded nodules can be removed from the false
positive count. Unmatched candidates are false positives.

The IOU boxes are converted from center-size format to corner format:

```text
pred = [x - w/2, x + w/2, y - h/2, y + h/2, z - d/2, z + d/2]
gt   = [x - w/2, x + w/2, y - h/2, y + h/2, z - d/2, z + d/2]
```

### FROC Vectors

The evaluator builds three lists:

```text
FROCGTList   = 1 for detected/missed nodules, 0 for false positive candidates
FROCProbList = candidate probability, or a very low sentinel for missed nodules
excludeList  = whether a sample should be excluded from ROC/FROC construction
```

Missed included nodules are appended with:

```text
ground_truth = 1
probability = -1000000000.0
exclude = True
```

This keeps the denominator of total nodules correct while preventing the
sentinel from behaving like a real candidate.

`computeFROC(...)` then computes an ROC curve over candidate probabilities and
converts false-positive rate into average false positives per scan:

```text
FPs_per_scan = fpr * (number_of_candidates - number_of_detected_lesions)
             / number_of_scans

sensitivity = tpr * number_of_detected_lesions / total_number_of_lesions
```

### CPM / Mean FROC Score

The code reports sensitivity at the standard LUNA-style false-positive rates:

```text
[0.125, 0.25, 0.5, 1, 2, 4, 8] false positives per scan
```

The mean FROC score, also commonly called CPM in this context, is:

```text
CPM = mean(sensitivity at [0.125, 0.25, 0.5, 1, 2, 4, 8] FP/scan)
```

In this repository, bootstrapping is enabled in the evaluation script:

```python
bPerformBootstrapping = True
bNumberOfBootstrapSamples = 1000
bConfidence = 0.95
```

The bootstrapped mean sensitivity curve is used to extract the seven reported
FROC points. Lightning logs:

```text
test/mean_froc
test/froc_0.125fp
test/froc_0.25fp
test/froc_0.5fp
test/froc_1fp
test/froc_2fp
test/froc_4fp
test/froc_8fp
```

The evaluator also writes text/CSV/PNG artifacts under:

```text
predictions/test_froc_epoch_<epoch>/
```

including:

- `CADAnalysis_<iou_threshold>.txt`;
- `froc_<prediction_file>_<iou_threshold>.txt`;
- `froc_gt_prob_vectors_<prediction_file>_<iou_threshold>.csv`;
- `froc_<prediction_file>_bootstrapping_<iou_threshold>.csv`;
- `froc_<prediction_file>_<iou_threshold>.png`.

## Training

The Hydra entrypoint is:

```bash
python -m src.det.CPMNetv2.train_lightning
```

The convenience launcher is:

```bash
bash bash/cpmnetv2_train_lightning_hydra.sh
```

Common overrides:

```bash
python -m src.det.CPMNetv2.train_lightning \
  csv_path=data/dataset_nodule_mean.csv \
  images_dir=/ssd2/domenico/datasets/lidc_process \
  annotations_dir=/ssd2/domenico/datasets/lidc_process \
  labels_csv=/ssd2/domenico/datasets/lidc_process/lidc_labels.csv \
  output_dir=outputs/cpmnetv2 \
  experiment_name=cpmnetv2 \
  max_epochs=150 \
  accelerator=gpu \
  devices='[0]' \
  precision=bf16-mixed \
  batch_size=2 \
  num_samples=10 \
  accumulate_grad_batches=8 \
  spacing='[1.0, 1.0, 1.0]'
```

Lightning saves checkpoints in:

```text
outputs/cpmnetv2/<experiment_name>/checkpoints/
```

The checkpoint monitor is validation loss:

```text
monitor = val/loss
mode = min
```

## Test-Only Evaluation

To run inference and FROC from a checkpoint:

```bash
python -m src.det.CPMNetv2.train_lightning \
  test_only=true \
  checkpoint='outputs/cpmnetv2/cpmnetv2/checkpoints/last.ckpt' \
  csv_path=data/dataset_nodule_mean.csv \
  images_dir=/ssd2/domenico/datasets/lidc_process \
  annotations_dir=/ssd2/domenico/datasets/lidc_process \
  labels_csv=/ssd2/domenico/datasets/lidc_process/lidc_labels.csv \
  no_froc=false \
  froc_iou_threshold=0.1
```

Disable FROC, for example when you only need prediction CSVs:

```bash
python -m src.det.CPMNetv2.train_lightning \
  test_only=true \
  checkpoint='outputs/cpmnetv2/cpmnetv2/checkpoints/last.ckpt' \
  no_froc=true
```

## Practical Notes

- Coordinate order matters. Model internals are `(z, y, x, d, h, w)`, while
  FROC CSV files are `(x, y, z, w, h, d)`.
- `spacing` affects positive center assignment. Use spacing that matches the
  physical voxel spacing of the preprocessed data.
- `topk` controls how many center points are assigned to each nodule. Larger
  values increase positive supervision but can make assignment less selective.
- `post_threshold`, `post_nms_threshold`, and `final_nms_overlap` directly affect
  FROC. A lower threshold increases sensitivity but may raise false positives.
- FROC bootstrapping uses 1000 samples and can be slow on large test sets.
- Validation in the current Lightning wrapper logs crop-level loss only.
  Full-volume split-combine prediction and FROC are performed in `test_step`.

## Citation

If you use CPMNet/CPMNetv2, cite the original papers:

```bibtex
@inproceedings{song2020cpm,
  title={CPM-Net: A 3D Center-Points Matching Network for Pulmonary Nodule Detection in CT Scans},
  author={Song, Tao and Chen, Jieneng and Luo, Xiangde and Huang, Yechong and Liu, Xinglong and Huang, Ning and Chen, Yinan and Ye, Zhaoxiang and Sheng, Huaqiang and Zhang, Shaoting and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={550--559},
  year={2020},
  organization={Springer}
}

@article{luo2021scpmnet,
  title={SCPM-Net: An anchor-free 3D lung nodule detection network using sphere representation and center points matching},
  author={Luo, Xiangde and Song, Tao and Wang, Guotai and Chen, Jieneng and Chen, Yinan and Li, Kang and Metaxas, Dimitris N and Zhang, Shaoting},
  journal={Medical Image Analysis},
  volume={75},
  pages={102287},
  year={2022},
  publisher={Elsevier}
}
```
