#!/usr/bin/env bash
# Foundation-model baseline: 3DINO-ViT, final 12 blocks fine-tuned, mean+max
# pooling over overlapping 112^3 windows. Requires the official teacher checkpoint.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

# PCI_BUS_ID makes CUDA_VISIBLE_DEVICES follow the nvidia-smi GPU indices.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XFORMERS_DISABLED=1

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
    python -m src.luna16_3dino_3d.train \
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
  python -m src.luna16_3dino_3d.aggregate \
    --output-dir "${OUTPUT_ROOT}" \
    --report-name "dino3d_pooled_results.md"

  cp "${OUTPUT_ROOT}/dino3d_pooled_results.md" "${RESULTS_DIR}/dino3d_pooled_results.md"
  cp "${OUTPUT_ROOT}/pooled_metrics.csv" "${RESULTS_DIR}/pooled_metrics.csv"
  cp "${OUTPUT_ROOT}/per_fold_mean_std_metrics.csv" "${RESULTS_DIR}/per_fold_mean_std_metrics.csv"

  echo "Done. Report copied to ${RESULTS_DIR}/dino3d_pooled_results.md"
} 2>&1 | tee "${LOG_FILE}"
