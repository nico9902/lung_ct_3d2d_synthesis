#!/usr/bin/env bash
# Volumetric baseline: 3D ResNet18 on the whole preprocessed CT volume.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/luna16_volume_3d_resnet18}"
RESULTS_DIR="${RESULTS_DIR:-results}"
VOLUME_D="${VOLUME_D:-160}"
VOLUME_H="${VOLUME_H:-224}"
VOLUME_W="${VOLUME_W:-224}"
NO_RESIZE="${NO_RESIZE:-0}"
if [[ "${NO_RESIZE}" == "1" ]]; then
  CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/cache_resnet18_native}"
else
  CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/cache_resnet18_${VOLUME_D}x${VOLUME_H}x${VOLUME_W}}"
fi

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-2}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-16-mixed}"
WANDB_PROJECT="${WANDB_PROJECT:-luna16-volume-3d}"
WANDB_GROUP="${WANDB_GROUP:-resnet18_fullvol_96x160x160}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"

mkdir -p logs/luna16_volume_3d "${OUTPUT_DIR}" "${RESULTS_DIR}" "${CACHE_DIR}"
LOG_FILE="logs/luna16_volume_3d/$(date +%Y%m%d_%H%M%S)_resnet18_10fold.log"

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "no_resize=${NO_RESIZE}, volume_size=${VOLUME_D} ${VOLUME_H} ${VOLUME_W}, batch_size=${BATCH_SIZE}, accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}, epochs=${EPOCHS}, lr=${LR}, precision=${PRECISION}"
echo "wandb_project=${WANDB_PROJECT}, wandb_group=${WANDB_GROUP}, wandb_offline=${WANDB_OFFLINE}"

WANDB_ARGS=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}")
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ "${WANDB_OFFLINE}" == "1" ]]; then
  WANDB_ARGS+=(--wandb-offline)
fi

for FOLD in ${FOLDS}; do
  echo "===== fold ${FOLD} ====="
  SIZE_ARGS=(--volume-size "${VOLUME_D}" "${VOLUME_H}" "${VOLUME_W}")
  if [[ "${NO_RESIZE}" == "1" ]]; then
    SIZE_ARGS+=(--no-resize)
  fi
  python -m src.luna16_volume_3d.train_resnet18 \
    --data-root "${DATA_ROOT}" \
    --splits-dir "${SPLITS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --fold "${FOLD}" \
    "${SIZE_ARGS[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --accumulate-grad-batches "${ACCUMULATE_GRAD_BATCHES}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --num-workers "${NUM_WORKERS}" \
    --precision "${PRECISION}" \
    --cache-dir "${CACHE_DIR}" \
    --monitor val_mcc \
    --no-early-stopping \
    --wandb-name "fold_${FOLD}_resnet18_3d" \
    "${WANDB_ARGS[@]}"
done 2>&1 | tee "${LOG_FILE}"

python -m src.luna16_volume_3d.aggregate_resnet18 \
  --output-dir "${OUTPUT_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --name "luna16_volume_3d_resnet18_pooled_results"
