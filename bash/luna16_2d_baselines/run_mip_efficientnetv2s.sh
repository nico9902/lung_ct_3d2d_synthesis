#!/usr/bin/env bash
set -euo pipefail

cd /home/domenico/lung_ct_3d2d_synthesis
source myenv/bin/activate

export PYTHONPATH="/home/domenico/lung_ct_3d2d_synthesis:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/ssd2/domenico/datasets/LUNA16_preprocessed}"
SPLITS_DIR="${SPLITS_DIR:-/ssd2/domenico/datasets/LUNA16_preprocessed/cv_splits}"
BASELINE_ROOT="${BASELINE_ROOT:-/ssd2/domenico/datasets/2d_baselines}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-256}"
IMAGE_WIDTH="${IMAGE_WIDTH:-384}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
DEVICES="${DEVICES:-[0]}"
PRECISION="${PRECISION:-32}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
WANDB_PROJECT="${WANDB_PROJECT:-luna16-2d-baselines-mip}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
EXPORT_BACKBONE_SUMMARY="${EXPORT_BACKBONE_SUMMARY:-0}"
MODES="${MODES:-axial triview}"

mkdir -p logs/luna16_2d_baselines

for MODE in ${MODES}; do
  if [[ "${MODE}" == central_* ]]; then
    BASELINE_KIND="central_slice"
  else
    BASELINE_KIND="mip"
  fi

  DATASET_DIR="${BASELINE_ROOT}/luna16_${BASELINE_KIND}_${MODE}_${IMAGE_HEIGHT}x${IMAGE_WIDTH}"
  echo "Generating non-adaptive dataset: ${MODE} -> ${DATASET_DIR}"
  python3 -m src.luna16_synthetic_2d.generate_mip_baselines \
    --data-root "${DATA_ROOT}" \
    --split-csv "${SPLITS_DIR}/luna16_classification_fold0.csv" \
    --output-dir "${DATASET_DIR}" \
    --mode "${MODE}" \
    --image-size "${IMAGE_HEIGHT}" "${IMAGE_WIDTH}"

  EXPERIMENT_NAME="${BASELINE_KIND}_${MODE}_${IMAGE_HEIGHT}x${IMAGE_WIDTH}_efficientnet_v2_s"
  OUTPUT_DIR="outputs/luna16_2d_baseline_${EXPERIMENT_NAME}"
  LOG_FILE="logs/luna16_2d_baselines/$(date +%Y%m%d_%H%M%S)_${EXPERIMENT_NAME}.log"

  echo "Training ${EXPERIMENT_NAME}; logging to ${LOG_FILE}"
  WANDB=1 \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_GROUP_SUFFIX="${EXPERIMENT_NAME}" \
  WANDB_OFFLINE="${WANDB_OFFLINE}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  SYNTHETIC_IMAGES_DIR="${DATASET_DIR}" \
  SPLITS_DIR="${SPLITS_DIR}" \
  FOLDS="${FOLDS}" \
  BACKBONES="efficientnet_v2_s" \
  EPOCHS="${EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  PRECISION="${PRECISION}" \
  IMAGE_HEIGHT="${IMAGE_HEIGHT}" \
  IMAGE_WIDTH="${IMAGE_WIDTH}" \
  ACCELERATOR="gpu" \
  DEVICES="${DEVICES}" \
  ACCUMULATE_GRAD_BATCHES=1 \
  MONITOR="val_mcc" \
  EXPORT_BACKBONE_SUMMARY="${EXPORT_BACKBONE_SUMMARY}" \
  bash src/luna16_synthetic_2d/run_backbones_det_experiment.sh \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --num-workers "${NUM_WORKERS}" \
    2>&1 | tee "${LOG_FILE}"
done
