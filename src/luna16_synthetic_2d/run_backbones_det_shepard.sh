#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

VENV_PATH="${VENV_PATH:-myenv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs/luna16_synthetic_2d_top7_minprob0.3_shepard}"
SYNTHETIC_IMAGES_DIR="${SYNTHETIC_IMAGES_DIR:-data/luna16_saliency_synthetic_detector_top7_minprob0.3_shepard}"
SPLITS_DIR="${SPLITS_DIR:-/ssd2/domenico/datasets/LUNA16_preprocessed/cv_splits}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
BACKBONES="${BACKBONES:-vgg16 efficientnet_b0 efficientnet_b1 efficientnet_v2_s resnet18 resnet50 densenet121}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PRECISION="${PRECISION:-32}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-256}"
IMAGE_WIDTH="${IMAGE_WIDTH:-384}"
ACCELERATOR="${ACCELERATOR:-auto}"
DEVICES="${DEVICES:-[2]}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
MONITOR="${MONITOR:-val_mcc}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-luna16-synthetic-2d-detector-top7-minprob0.3-shepard}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-0}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-0}"
FREEZE_HALF_BACKBONE="${FREEZE_HALF_BACKBONE:-0}"
FREEZE_FIRST_LAYERS="${FREEZE_FIRST_LAYERS:-0}"
UNFREEZE_LAST_LAYERS="${UNFREEZE_LAST_LAYERS:-0}"
EXPORT_BACKBONE_SUMMARY="${EXPORT_BACKBONE_SUMMARY:-1}"
BACKBONE_SUMMARY_XLSX="${BACKBONE_SUMMARY_XLSX:-backbone_summary.xlsx}"

EXTRA_ARGS=()
if [[ "${WANDB}" == "1" ]]; then
  EXTRA_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  EXTRA_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ "${WANDB_OFFLINE}" == "1" ]]; then
  EXTRA_ARGS+=(--wandb-offline)
fi
if [[ "${WANDB_LOG_MODEL}" == "1" ]]; then
  EXTRA_ARGS+=(--wandb-log-model)
fi
if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
  EXTRA_ARGS+=(--freeze-backbone)
fi

if [[ "${FREEZE_HALF_BACKBONE}" == "1" ]]; then
  EXTRA_ARGS+=(--freeze-half-backbone)
fi

if [[ "${FREEZE_FIRST_LAYERS}" != "0" ]]; then
  EXTRA_ARGS+=(--freeze-first-layers "${FREEZE_FIRST_LAYERS}")
fi

if [[ "${UNFREEZE_LAST_LAYERS}" != "0" ]]; then
  EXTRA_ARGS+=(--unfreeze-last-layers "${UNFREEZE_LAST_LAYERS}")
fi

for FOLD in ${FOLDS}; do
  for BACKBONE in ${BACKBONES}; do
    WANDB_RUN_GROUP="${BACKBONE}" python3 -m src.luna16_synthetic_2d.train \
      --output-dir "${OUTPUT_DIR}" \
      --synthetic-images-dir "${SYNTHETIC_IMAGES_DIR}" \
      --splits-dir "${SPLITS_DIR}" \
      --fold "${FOLD}" \
      --backbone "${BACKBONE}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --image-size "${IMAGE_HEIGHT}" "${IMAGE_WIDTH}" \
      --precision "${PRECISION}" \
      --accelerator "${ACCELERATOR}" \
      --devices "${DEVICES}" \
      --accumulate-grad-batches "${ACCUMULATE_GRAD_BATCHES}" \
      --monitor "${MONITOR}" \
      --wandb-group "${BACKBONE}_top5_minprob0.5" \
      "${EXTRA_ARGS[@]}"
  done
done

if [[ "${EXPORT_BACKBONE_SUMMARY}" == "1" ]]; then
  python3 -m src.luna16_synthetic_2d.export_backbone_summary \
    --output-dir "${OUTPUT_DIR}" \
    --xlsx-name "${BACKBONE_SUMMARY_XLSX}"
fi
