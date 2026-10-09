#!/usr/bin/env bash
# Detector-top axial slice comparator using the central-slice training setup.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

IMAGE_HEIGHT="${IMAGE_HEIGHT:-256}"
IMAGE_WIDTH="${IMAGE_WIDTH:-384}"
DATASET_DIR="${DATASET_DIR:-outputs/luna16_2d_baseline_detector_top_axial_${IMAGE_HEIGHT}x${IMAGE_WIDTH}_images}"
PRED_ROOT="${PRED_ROOT:-${DETECTOR_PREDICTIONS}}"
EXPERIMENT_NAME="detector_top_axial_${IMAGE_HEIGHT}x${IMAGE_WIDTH}_efficientnet_v2_s"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/luna16_2d_baseline_${EXPERIMENT_NAME}}"

python -m src.luna16_synthetic_2d.generate_detector_top_slice_baseline \
  --data-root "${DATA_ROOT}" \
  --splits-dir "${SPLITS_DIR}" \
  --pred-root "${PRED_ROOT}" \
  --output-dir "${DATASET_DIR}" \
  --image-size "${IMAGE_HEIGHT}" "${IMAGE_WIDTH}"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Prepared detector-top slice images at ${DATASET_DIR}"
  exit 0
fi

mkdir -p logs/luna16_2d_baselines
LOG_FILE="logs/luna16_2d_baselines/$(date +%Y%m%d_%H%M%S)_${EXPERIMENT_NAME}.log"

WANDB="${WANDB:-1}" \
WANDB_PROJECT="${WANDB_PROJECT:-luna16-2d-baselines-nonadaptive}" \
WANDB_GROUP_SUFFIX="${EXPERIMENT_NAME}" \
WANDB_OFFLINE="${WANDB_OFFLINE:-0}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
SYNTHETIC_IMAGES_DIR="${DATASET_DIR}" \
SPLITS_DIR="${SPLITS_DIR}" \
FOLDS="${FOLDS}" \
BACKBONES="efficientnet_v2_s" \
EPOCHS="${EPOCHS:-100}" \
BATCH_SIZE="${BATCH_SIZE:-16}" \
PRECISION="${PRECISION:-32}" \
IMAGE_HEIGHT="${IMAGE_HEIGHT}" \
IMAGE_WIDTH="${IMAGE_WIDTH}" \
ACCELERATOR="${ACCELERATOR:-gpu}" \
DEVICES="${DEVICES:-[0]}" \
ACCUMULATE_GRAD_BATCHES=1 \
MONITOR=val_mcc \
EXPORT_BACKBONE_SUMMARY="${EXPORT_BACKBONE_SUMMARY:-0}" \
bash "${PROJECT_DIR}/bash/luna16_synthetic_2d/train_backbones.sh" \
  --lr "${LR:-0.0001}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --num-workers "${NUM_WORKERS:-4}" \
  2>&1 | tee "${LOG_FILE}"
