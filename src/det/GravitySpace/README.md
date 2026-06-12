# GravitySpace Subvolume Training

This module can train either on full CT volumes or on fixed-depth subvolumes.
Subvolumes reduce GPU memory because the model receives fewer slices at once:

```text
full volume: B x S x C x H x W
subvolume:   B x S_sub x C x H x W
```

The model code keeps the same input contract. The dataset now crops the slice
dimension before transforms, so `GravitySpaceAttentionNet` still sees a normal
5D tensor.

## Main Files

- `dataset_LIDC_IDRI.py`
  Builds subvolume windows and returns cropped samples.
- `datamodule.py`
  Passes subvolume options to train/val/test datasets and preserves
  `original_lengths` for padded tail windows.
- `conf/data/lidc_idri.yaml`
  Contains the Hydra options used to enable or disable subvolumes.
- `train.py`
  Reads the Hydra options and constructs the datamodule.

## Configuration

Subvolumes are disabled by default:

```yaml
use_subvolumes: false
subvolume_depth: 32
subvolume_stride: 16
val_subvolume_stride: null
test_subvolume_stride: null
positive_fraction: 0.7
samples_per_epoch: null
```

Enable them from the command line:

```bash
data.use_subvolumes=true \
data.subvolume_depth=32 \
data.subvolume_stride=16 \
data.val_subvolume_stride=16 \
data.test_subvolume_stride=16 \
data.positive_fraction=0.7 \
data.samples_per_epoch=2000
```

Recommended first memory-saving setup:

```bash
data.batch_size=1 \
data.use_subvolumes=true \
data.subvolume_depth=32 \
data.subvolume_stride=16 \
data.val_subvolume_stride=16 \
data.test_subvolume_stride=16 \
data.positive_fraction=0.7 \
model.chunk_size=10 \
model.window_size=5 \
model.sampling=1
```

## How Windows Are Built

For each case, the dataset reads the annotation volume or precomputed annotation
centers and creates windows along the slice dimension.

Training windows use `subvolume_stride`, so they can overlap:

```text
depth = 32
stride = 16

0:32, 16:48, 32:64, ...
```

Validation windows are deterministic and non-overlapping by default:

```text
0:32, 32:64, 64:96, ...
```

If `val_subvolume_stride` or `test_subvolume_stride` is set, eval windows
overlap for extra context:

```text
depth = 32
eval stride = 16

input windows: 0:32, 16:48, 32:64, ...
```

Only part of each overlapping validation/test window is counted for loss,
detections, and FROC:

```text
window 0:32  evaluates global slices 0:16
window 16:48 evaluates global slices 16:32
window 32:64 evaluates global slices 32:48
```

This gives border slices more cross-slice context without counting the same
slice twice. Validation loss also uses these eval ranges, so overlap does not
inflate `val/loss`.

If the final window is shorter than `subvolume_depth`, it is padded. The real
length is carried through `original_lengths`, so the loss and inference loops
ignore padded slices.

## Positive/Negative Balance

A subvolume is considered positive when it contains at least one nodule:

- With precomputed centers: any `annotations[:, :, 0] != -1`
- With raw masks: any voxel/pixel value greater than zero

During training, `__getitem__` samples:

```text
positive_fraction from positive windows
1 - positive_fraction from negative windows
```

For example, `positive_fraction=0.7` means roughly 70% positive subvolumes and
30% background subvolumes. This prevents training from being dominated by empty
lung slices while still showing the detector enough negatives.

If a split has no positives or no negatives, the sampler falls back to whichever
pool is available.

## Returned Sample

With subvolumes enabled, `__getitem__` returns the same main keys as before:

```python
{
    "case": case,
    "slices": slices,              # [S_sub, H, W] before transforms
    "annotations": annotations,    # [S_sub, max_nodules, 4] for centers
    "slicenames": slicenames,
}
```

It also includes metadata:

```python
{
    "subvolume_start": start,
    "subvolume_end": end,
    "eval_slice_start": local_eval_start,
    "eval_slice_end": local_eval_end,
    "has_nodule": True or False,
    "original_length": real_length_before_padding,
}
```

The collate function converts `original_length` into `original_lengths`, which
is already consumed by the loss and inference code. The `eval_slice_start` and
`eval_slice_end` fields are used during validation/test/inference to avoid
duplicate supervision or detections when overlapping eval windows are enabled.

## Subvolume Bash Script

Use the dedicated launch script for the memory-optimized subvolume setup:

```bash
bash bash/gravity_space_detection_subvolumes.sh
```

The script enables train subvolume sampling, overlapping validation/test
context, `batch_size=1`, and backbone chunking.

## Important Caveat

Keep `model.sampling=1` unless the loss is updated to account for sampled slice
indices. The current loss indexes predictions using the original slice index,
so `sampling > 1` can make prediction length shorter than annotation length.

Subvolumes reduce memory by reducing `S`; `sampling=1` preserves the current
loss behavior.
