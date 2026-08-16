#!/usr/bin/env bash
set -euo pipefail

cd /home/domenico/lung_ct_3d2d_synthesis
source myenv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/home/domenico/lung_ct_3d2d_synthesis:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/ssd2/domenico/datasets/LUNA16_preprocessed}"
SPLITS_DIR="${SPLITS_DIR:-/ssd2/domenico/datasets/LUNA16_preprocessed/cv_splits}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/luna16_volume_3d_resnet18}"
RESULTS_DIR="${RESULTS_DIR:-results}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-16-mixed}"
VOLUME_D="${VOLUME_D:-96}"
VOLUME_H="${VOLUME_H:-160}"
VOLUME_W="${VOLUME_W:-160}"

mkdir -p logs/luna16_volume_3d "${OUTPUT_DIR}" "${RESULTS_DIR}"
LOG_FILE="logs/luna16_volume_3d/$(date +%Y%m%d_%H%M%S)_resnet18_10fold.log"

echo "Logging to ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "volume_size=${VOLUME_D} ${VOLUME_H} ${VOLUME_W}, batch_size=${BATCH_SIZE}, epochs=${EPOCHS}, lr=${LR}, precision=${PRECISION}"

for FOLD in 0 1 2 3 4 5 6 7 8 9; do
  echo "===== fold ${FOLD} ====="
  python src/luna16_volume_3d/train_resnet18.py \
    --data-root "${DATA_ROOT}" \
    --splits-dir "${SPLITS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --fold "${FOLD}" \
    --volume-size "${VOLUME_D}" "${VOLUME_H}" "${VOLUME_W}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --num-workers "${NUM_WORKERS}" \
    --precision "${PRECISION}" \
    --monitor val_mcc
done 2>&1 | tee "${LOG_FILE}"

python src/luna16_volume_3d/aggregate_resnet18.py \
  --output-dir "${OUTPUT_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --name "luna16_volume_3d_resnet18_pooled_results"
