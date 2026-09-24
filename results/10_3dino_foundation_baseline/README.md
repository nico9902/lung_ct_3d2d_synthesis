# 3DINO-ViT Partial-Fine-Tuning Foundation-Model Baseline

Compares the Adaptive RBF approach against a **partially fine-tuned** [3DINO-ViT](https://github.com/AICONSlab/3DINO)
(Xu et al., npj Digital Medicine 2025) 3D CT foundation model for LUNA16 patient-level benign-vs-malignant
classification, using the exact same patient labels and 10-fold splits as the rest of the project.

Unlike COLIPRI/Rad-JEPA-3D (fully frozen encoder + linear probe), the first 12 of 24 ViT-Large-3D transformer
blocks are frozen and the second 12 blocks + final norm + a new mean+max-pooling classification head are trained
with discriminative learning rates. No lesion-guided information (CPMNetv2 detections, nodule coordinates, masks)
is used -- only the full CT volume, split into overlapping 112^3 sliding windows.

Code: [`src/luna16_3dino_3d/`](../../src/luna16_3dino_3d/). Reproduce with:

```bash
CHECKPOINT_PATH=/path/to/3dino_vit_weights.pth CUDA_VISIBLE_DEVICES=3 bash bash/luna16_3dino_3d/run_3dino_baseline.sh
```

See the main [README](../../README.md#3dino-partial-fine-tuning-foundation-model-baseline) for the full pipeline
description and the exact official-source verification of every architectural/preprocessing detail (native input
size, global representation, checkpoint loading, block-freezing under the checkpoint's `block_chunks=4` layout).

## Status

`completed` -- the official 3DINO-ViT checkpoint was evaluated across all 10 held-out folds. The pooled results are
reported in `dino3d_pooled_results.md`, `pooled_metrics.csv`, and `per_fold_mean_std_metrics.csv`. The checkpoint is
gated on Hugging Face
([AICONSlab/3DINO-ViT](https://huggingface.co/AICONSlab/3DINO-ViT), CC BY-NC-ND 4.0) and requires accepting the
license under a personal account before it can be downloaded. The full pipeline (data module, partial-freeze
Lightning module, training script, wandb integration, aggregation) was run to completion on the real checkpoint.

Compute note: partial fine-tuning (not just a frozen-feature linear probe) over full-volume sliding windows is
substantially more expensive than the COLIPRI/Rad-JEPA-3D baselines. With preprocessed windows cached to disk
(shared across all 10 folds), measured per-patient cost drops from ~3.5s (cold, CPU-bound: CT loading + windowing
+ percentile normalization) to a GPU-bound ~0.5-0.9s (~80 windows/patient on an A100, bf16). A full 10-fold run is
estimated at roughly 1-2 days of A100 time.
