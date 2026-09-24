#!/usr/bin/env bash
set -euo pipefail

cd /home/domenico/lung_ct_3d2d_synthesis
source myenv/bin/activate

# 3DINO-ViT baseline: use the A100 (physical GPU index 3 on this host, per
# `nvidia-smi`'s PCI-bus-ID ordering) as the sole visible device, so it
# appears as cuda:0 ("DEVICE 0"). CUDA_DEVICE_ORDER=PCI_BUS_ID is required:
# PyTorch's default CUDA enumeration does NOT match nvidia-smi's indices on
# this host (index 3 without it silently resolves to a V100).
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONPATH="/home/domenico/lung_ct_3d2d_synthesis:${PYTHONPATH:-}"
export XFORMERS_DISABLED=1

DATA_ROOT="${DATA_ROOT:-data/processed}"
SPLITS_DIR="${SPLITS_DIR:-data/processed/cv_splits}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/luna16_3dino_3d}"
WINDOWS_CACHE_DIR="${WINDOWS_CACHE_DIR:-${OUTPUT_ROOT}/window_cache}"
RESULTS_DIR="${RESULTS_DIR:-results/10_3dino_foundation_baseline}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the official 3DINO-ViT teacher checkpoint (.pth)}"

WINDOW_SIZE="${WINDOW_SIZE:-112}"
STRIDE="${STRIDE:-56}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
HEAD_LR="${HEAD_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
EPOCHS="${EPOCHS:-50}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
PRECISION="${PRECISION:-bf16-true}"
MONITOR="${MONITOR:-val_auc}"
PATIENCE="${PATIENCE:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MICRO_BATCH_WINDOWS="${MICRO_BATCH_WINDOWS:-32}"
SEED="${SEED:-233}"
WANDB_PROJECT="${WANDB_PROJECT:-luna16-3dino-3d}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"

mkdir -p logs/luna16_3dino_3d "${OUTPUT_ROOT}" "${WINDOWS_CACHE_DIR}" "${RESULTS_DIR}"
LOG_FILE="logs/luna16_3dino_3d/$(date +%Y%m%d_%H%M%S)_3dino_baseline.log"

WANDB_ARGS=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi
  if [[ "${WANDB_OFFLINE}" == "1" ]]; then
    WANDB_ARGS+=(--wandb-offline)
  fi
fi

{
  echo "Logging to ${LOG_FILE}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER})"
  echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
  echo "WINDOWS_CACHE_DIR=${WINDOWS_CACHE_DIR} (shared across all folds -- extracted once)"

  for FOLD in ${FOLDS}; do
    echo "===== fold ${FOLD} ====="
    python src/luna16_3dino_3d/train.py \
      --data-root "${DATA_ROOT}" \
      --splits-dir "${SPLITS_DIR}" \
      --output-dir "${OUTPUT_ROOT}" \
      --windows-cache-dir "${WINDOWS_CACHE_DIR}" \
      --checkpoint-path "${CHECKPOINT_PATH}" \
      --fold "${FOLD}" \
      --window-size "${WINDOW_SIZE}" \
      --stride "${STRIDE}" \
      --backbone-lr "${BACKBONE_LR}" \
      --head-lr "${HEAD_LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --warmup-epochs "${WARMUP_EPOCHS}" \
      --epochs "${EPOCHS}" \
      --accumulate-grad-batches "${ACCUMULATE_GRAD_BATCHES}" \
      --gradient-clip-val "${GRADIENT_CLIP_VAL}" \
      --precision "${PRECISION}" \
      --monitor "${MONITOR}" \
      --patience "${PATIENCE}" \
      --num-workers "${NUM_WORKERS}" \
      --micro-batch-windows "${MICRO_BATCH_WINDOWS}" \
      --seed "${SEED}" \
      "${WANDB_ARGS[@]}"
  done

  echo "===== Aggregating pooled + per-fold mean/std metrics ====="
  python src/luna16_3dino_3d/aggregate.py \
    --output-dir "${OUTPUT_ROOT}" \
    --report-name "dino3d_pooled_results.md"

  cp "${OUTPUT_ROOT}/dino3d_pooled_results.md" "${RESULTS_DIR}/dino3d_pooled_results.md"
  cp "${OUTPUT_ROOT}/pooled_metrics.csv" "${RESULTS_DIR}/pooled_metrics.csv"
  cp "${OUTPUT_ROOT}/per_fold_mean_std_metrics.csv" "${RESULTS_DIR}/per_fold_mean_std_metrics.csv"

  echo "Done. Report copied to ${RESULTS_DIR}/dino3d_pooled_results.md"
} 2>&1 | tee "${LOG_FILE}"
