# Rad-JEPA-3D Frozen Foundation-Model Baseline

Compares the Adaptive RBF approach against a second **frozen** 3D CT foundation model, Rad-JEPA-3D (an H-Mamba
hybrid encoder; checkpoint and code reused by absolute path from the sibling project
`/home/domenico/lung-jepa-world-model`), for LUNA16 patient-level benign-vs-malignant classification, using the
exact same patient labels and 10-fold splits as the rest of the project
(`data/processed/cv_splits/luna16_classification_fold{0-9}.csv`).

No lesion-guided information (CPMNetv2 detections, nodule coordinates, masks) is used — Rad-JEPA only ever sees the
full CT volume, representing a generic full-volume foundation-model baseline, same spirit as COLIPRI
([`results/08_colipri_foundation_baseline`](../08_colipri_foundation_baseline)).

Code: [`src/luna16_radjepa_3d/`](../../src/luna16_radjepa_3d/). Reproduce with:

```bash
CUDA_VISIBLE_DEVICES=3 bash bash/luna16_radjepa_3d/run_radjepa_baseline.sh
```

See the main [README](../../README.md#rad-jepa-3d-foundation-model-baseline) for the full pipeline description.

Four patient-level representations are compared, all derived from the same cached per-window embeddings
(the encoder is run once per patient, never re-run to compare strategies):

- **resize32** (384-D): whole scan resized to `(32, 256, 256)`.
- **sliding_mean** (384-D): mean of 32-slice/stride-16 sliding-window embeddings.
- **sliding_max** (384-D): element-wise max of the same windows.
- **sliding_mean_max** (768-D): concatenation of the mean and max above.

## Status

`complete` — run end-to-end on the full 888-patient cohort (796 binary benign/malignant patients across the 10
folds) on an NVIDIA A100 80GB. Full report: [`radjepa_pooled_results.md`](radjepa_pooled_results.md); raw tables:
[`pooled_metrics.csv`](pooled_metrics.csv), [`per_fold_mean_std_metrics.csv`](per_fold_mean_std_metrics.csv).

### Pooled out-of-fold test metrics (all 10 folds concatenated, 796 samples)

| Representation | AUC | MCC | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| resize32 | 0.5679 | 0.1373 | 0.5842 | 0.4868 | 0.4831 | 0.4906 |
| sliding_mean | 0.5838 | 0.1504 | 0.5917 | 0.4914 | 0.4922 | 0.4906 |
| sliding_max | 0.5765 | 0.1498 | 0.5829 | 0.5103 | 0.4832 | 0.5406 |
| sliding_mean_max | **0.5934** | **0.1615** | 0.5854 | **0.5231** | 0.4866 | 0.5656 |

### Per-fold mean +/- std (ddof=1)

| Representation | AUC | MCC | Accuracy | F1 |
|---|---:|---:|---:|---:|
| resize32 | 0.5779 +/- 0.0528 | 0.1232 +/- 0.1058 | 0.5844 +/- 0.0407 | 0.4697 +/- 0.1196 |
| sliding_mean | 0.5873 +/- 0.0584 | 0.1365 +/- 0.1155 | 0.5919 +/- 0.0464 | 0.4750 +/- 0.1221 |
| sliding_max | 0.6031 +/- 0.0555 | 0.1428 +/- 0.1203 | 0.5833 +/- 0.0550 | 0.4977 +/- 0.1111 |
| sliding_mean_max | 0.6078 +/- 0.0500 | 0.1619 +/- 0.1017 | 0.5858 +/- 0.0540 | 0.5209 +/- 0.0639 |

## Interpretation

Sliding-window pooling consistently beats naively resizing the whole scan to 32 slices (`resize32`): both `AUC`
and `MCC` improve monotonically resize32 -> sliding_mean -> sliding_max -> sliding_mean_max, and concatenating
mean+max gives the best pooled AUC (0.5934) and MCC (0.1615). This is expected: `resize32` compresses the whole
z-extent (often 100-400+ native slices) down to 32 slices in one shot, discarding most through-plane detail a small
pulmonary nodule occupies, whereas the sliding windows preserve native depth resolution within each 32-slice
segment.

Compared against the other two baselines on the identical 796-patient pooled test set
([`results/00_paper_overview`](../00_paper_overview)):

| Method | AUC | MCC |
|---|---:|---:|
| Adaptive RBF, detector-guided | 0.8149 | 0.4780 |
| COLIPRI-CRM, pooled | 0.8508 | 0.5265 |
| Rad-JEPA-3D, sliding_mean_max (best) | 0.5934 | 0.1615 |

Rad-JEPA-3D trails both the Adaptive RBF pipeline and COLIPRI by a wide margin, only modestly above chance (AUC
0.5). Plausible explanations, none mutually exclusive: (1) Rad-JEPA-3D was pretrained on a different CT
distribution/task (its M3D-Cap pretraining data and JEPA masked-prediction objective are not tuned for
nodule-scale malignancy signal); (2) the frozen 384-D mean-pooled token embedding may not preserve the localized,
small-lesion detail needed here, since mean-pooling over 1024 spatial tokens can dilute a signal occupying only a
few tokens' worth of the volume; (3) unlike COLIPRI (which has a dedicated attention-pooling head trained
end-to-end for a pooled embedding), Rad-JEPA-3D's checkpoint only exposes the raw patch/token encoder, so we are
pooling with simple mean/max rather than a representation the model was ever trained to produce as a global
summary. This result supports the paper's broader claim that generic frozen 3D foundation-model features are not
a drop-in substitute for a task-guided pipeline on this dataset, and that COLIPRI-style joint image-text
pretraining (or full fine-tuning) may matter more than raw pretrained-encoder scale for this task.
