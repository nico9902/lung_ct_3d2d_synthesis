#!/usr/bin/env bash
# Paper detector: CPMNetv2 trained on the 10 LUNA16 folds (bf16, batch 8,
# 3 samples/volume, top-k 7, warmup + cosine LR). Folds with an existing best
# checkpoint are skipped, so the script can be resumed.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/cpmnetv2_luna16_10fold_bf16_guarded}"
BASE_TAG="${BASE_TAG:-20260809_cpmnetv2_luna16_10fold_bs8_numsam3_topk7_a100_lrbase0001_lrmax001_bf16_guarded}"
LOG_DIR="${LOG_DIR:-logs/cpmnetv2}"

mkdir -p "${LOG_DIR}"

for FOLD in ${FOLDS}; do
  SPLIT_CSV="${SPLITS_DIR}/luna16_fold${FOLD}.csv"
  EXPERIMENT_NAME="${BASE_TAG}_fold${FOLD}"
  FOLD_LOG="${LOG_DIR}/${EXPERIMENT_NAME}.log"

  if [[ -f "${OUTPUT_DIR}/${EXPERIMENT_NAME}/checkpoints/cpmnetv2_luna16_best.ckpt" ]]; then
    echo "Skipping fold ${FOLD}: checkpoint already exists."
    continue
  fi

  echo "Starting fold ${FOLD}: ${EXPERIMENT_NAME} (log: ${FOLD_LOG})"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  SPLIT_CSV="${SPLIT_CSV}" \
  LABELS_CSV="${SPLIT_CSV}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  BATCH_SIZE="${BATCH_SIZE:-8}" \
  NUM_SAMPLES="${NUM_SAMPLES:-3}" \
  TOPK="${TOPK:-7}" \
  MAX_EPOCHS="${MAX_EPOCHS:-170}" \
  VAL_FROC_START_EPOCH="${VAL_FROC_START_EPOCH:-100}" \
  VAL_FROC_BEFORE_START_EVERY_N_EPOCH="${VAL_FROC_BEFORE_START_EVERY_N_EPOCH:-100}" \
  CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-5}" \
  CHECKPOINT_START_EPOCH="${CHECKPOINT_START_EPOCH:-100}" \
  NUM_WORKERS="${NUM_WORKERS:-8}" \
  DEVICES="${DEVICES:-[0]}" \
  ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}" \
  bash "${PROJECT_DIR}/bash/cpmnetv2/train_fold.sh" \
    precision=bf16-mixed \
    accelerator=gpu \
    use_wandb=True \
    wandb_project=lung_ct_3d2d_synthesis_detection \
    wandb_name="${EXPERIMENT_NAME}" \
    lr=0.001 \
    warmup_multiplier=10.0 \
    warmup_epochs=2 \
    cosine_t_max=170 \
    eta_min=1e-6 \
    "$@" \
    > "${FOLD_LOG}" 2>&1

  echo "Finished fold ${FOLD}: ${EXPERIMENT_NAME}"
done
