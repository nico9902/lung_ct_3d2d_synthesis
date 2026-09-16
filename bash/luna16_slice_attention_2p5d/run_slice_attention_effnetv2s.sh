#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/myenv}"

cd "${PROJECT_DIR}"
source "${VENV_PATH}/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/ssd2/domenico/datasets/LUNA16_preprocessed}"
SPLITS_DIR="${SPLITS_DIR:-/ssd2/domenico/datasets/LUNA16_preprocessed/cv_splits}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/luna16_slice_attention_2p5d_all_slices_256x384_effnetv2s}"
RESULTS_DIR="${RESULTS_DIR:-results/03_2d_nonadaptive_baselines}"
CACHE_DIR="${CACHE_DIR:-/ssd2/domenico/datasets/LUNA16_preprocessed/cache_slice_attention_256x384}"

IMAGE_HEIGHT="${IMAGE_HEIGHT:-256}"
IMAGE_WIDTH="${IMAGE_WIDTH:-384}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
BACKBONE="${BACKBONE:-efficientnet_v2_s}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
SLICE_CHUNK_SIZE="${SLICE_CHUNK_SIZE:-16}"
ATTENTION_DIM="${ATTENTION_DIM:-256}"
DROPOUT="${DROPOUT:-0.2}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16-mixed}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
MONITOR="${MONITOR:-val_mcc}"
CACHE_READ_ONLY="${CACHE_READ_ONLY:-0}"
CACHE_WRITE_FIRST_EPOCH_ONLY="${CACHE_WRITE_FIRST_EPOCH_ONLY:-0}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-luna16-slice-attention-2p5d}"
WANDB_GROUP="${WANDB_GROUP:-${BACKBONE}_all_slices_${IMAGE_HEIGHT}x${IMAGE_WIDTH}}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
WANDB_NAME_SUFFIX="${WANDB_NAME_SUFFIX:-slice_attention}"
AGGREGATE_NAME="${AGGREGATE_NAME:-slice_attention_2p5d_pooled_results}"
LIMIT_TRAIN_SAMPLES="${LIMIT_TRAIN_SAMPLES:-}"
LIMIT_VAL_SAMPLES="${LIMIT_VAL_SAMPLES:-}"
LIMIT_TEST_SAMPLES="${LIMIT_TEST_SAMPLES:-}"
FREEZE_HALF_BACKBONE="${FREEZE_HALF_BACKBONE:-0}"

mkdir -p logs/luna16_slice_attention_2p5d "${OUTPUT_DIR}" "${RESULTS_DIR}"
if [[ "${CACHE_READ_ONLY}" != "1" ]]; then
  mkdir -p "${CACHE_DIR}"
fi
LOG_FILE="logs/luna16_slice_attention_2p5d/$(date +%Y%m%d_%H%M%S)_${BACKBONE}_10fold.log"

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SPLITS_DIR=${SPLITS_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "CACHE_READ_ONLY=${CACHE_READ_ONLY}"
echo "CACHE_WRITE_FIRST_EPOCH_ONLY=${CACHE_WRITE_FIRST_EPOCH_ONLY}"
echo "backbone=${BACKBONE}, freeze_half_backbone=${FREEZE_HALF_BACKBONE}, image_size=${IMAGE_HEIGHT} ${IMAGE_WIDTH}, batch_size=${BATCH_SIZE}, accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}, slice_chunk_size=${SLICE_CHUNK_SIZE}"
echo "epochs=${EPOCHS}, lr=${LR}, weight_decay=${WEIGHT_DECAY}, precision=${PRECISION}, accelerator=${ACCELERATOR}, devices=${DEVICES}, monitor=${MONITOR}"
echo "wandb=${WANDB}, wandb_project=${WANDB_PROJECT}, wandb_group=${WANDB_GROUP}, wandb_offline=${WANDB_OFFLINE}"

WANDB_ARGS=()
if [[ "${WANDB}" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ "${WANDB_OFFLINE}" == "1" ]]; then
  WANDB_ARGS+=(--wandb-offline)
fi

CACHE_ARGS=(--cache-dir "${CACHE_DIR}")
if [[ "${CACHE_READ_ONLY}" == "1" ]]; then
  CACHE_ARGS+=(--cache-read-only)
fi
if [[ "${CACHE_WRITE_FIRST_EPOCH_ONLY}" == "1" ]]; then
  CACHE_ARGS+=(--cache-write-first-epoch-only)
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT_TRAIN_SAMPLES}" ]]; then
  LIMIT_ARGS+=(--limit-train-samples "${LIMIT_TRAIN_SAMPLES}")
fi
if [[ -n "${LIMIT_VAL_SAMPLES}" ]]; then
  LIMIT_ARGS+=(--limit-val-samples "${LIMIT_VAL_SAMPLES}")
fi
if [[ -n "${LIMIT_TEST_SAMPLES}" ]]; then
  LIMIT_ARGS+=(--limit-test-samples "${LIMIT_TEST_SAMPLES}")
fi

FREEZE_ARGS=()
if [[ "${FREEZE_HALF_BACKBONE}" == "1" ]]; then
  FREEZE_ARGS+=(--freeze-half-backbone)
fi

for FOLD in ${FOLDS}; do
  echo "===== fold ${FOLD} ====="
  python -m src.luna16_slice_attention_2p5d.train \
    --data-root "${DATA_ROOT}" \
    --splits-dir "${SPLITS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --fold "${FOLD}" \
    --backbone "${BACKBONE}" \
    --image-size "${IMAGE_HEIGHT}" "${IMAGE_WIDTH}" \
    --batch-size "${BATCH_SIZE}" \
    --accumulate-grad-batches "${ACCUMULATE_GRAD_BATCHES}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --attention-dim "${ATTENTION_DIM}" \
    --dropout "${DROPOUT}" \
    --slice-chunk-size "${SLICE_CHUNK_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --precision "${PRECISION}" \
    "${CACHE_ARGS[@]}" \
    --monitor "${MONITOR}" \
    --no-early-stopping \
    --wandb-name "fold_${FOLD}_${BACKBONE}_${WANDB_NAME_SUFFIX}" \
    "${FREEZE_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${LIMIT_ARGS[@]}" \
    "$@"
done 2>&1 | tee "${LOG_FILE}"

python -m src.luna16_slice_attention_2p5d.aggregate \
  --output-dir "${OUTPUT_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --backbones "${BACKBONE}" \
  --name "${AGGREGATE_NAME}"
