#!/usr/bin/env bash
set -euo pipefail

cd /home/domenico/lung_ct_3d2d_synthesis
source myenv/bin/activate

export PYTHONPATH="/home/domenico/lung_ct_3d2d_synthesis:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/ssd2/domenico/datasets/LUNA16_preprocessed}"
SPLITS_DIR="${SPLITS_DIR:-/ssd2/domenico/datasets/LUNA16_preprocessed/cv_splits}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-outputs/cpmnetv2_luna16_10fold_bf16_guarded_results/normalized_predictions}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/luna16_detection_mil_cpmnetv2_top4_minprob0.50_effnetv2s}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
POOLINGS="${POOLINGS:-mean max attention}"
TOP_K="${TOP_K:-4}"
MIN_PROBABILITY="${MIN_PROBABILITY:-0.5}"
CROP_SIZE_MM="${CROP_SIZE_MM:-64}"
CROP_IMAGE_SIZE="${CROP_IMAGE_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-100}"
PRECISION="${PRECISION:-32}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-[0]}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-luna16-detection-mil}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"

mkdir -p logs/luna16_detection_mil

EXTRA_ARGS=()
if [[ "${WANDB}" == "1" ]]; then
  EXTRA_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}")
fi
if [[ "${WANDB_OFFLINE}" == "1" ]]; then
  EXTRA_ARGS+=(--wandb-offline)
fi

echo "Training detection-crop MIL baseline"
echo "  Output: ${OUTPUT_DIR}"
echo "  Predictions: ${PREDICTIONS_ROOT}"
echo "  Top-k/min-probability: ${TOP_K}/${MIN_PROBABILITY}"
echo "  Poolings: ${POOLINGS}"
echo "  Folds: ${FOLDS}"

for POOLING in ${POOLINGS}; do
  for FOLD in ${FOLDS}; do
    python3 -m src.luna16_detection_mil.train \
      --output-dir "${OUTPUT_DIR}" \
      --data-root "${DATA_ROOT}" \
      --splits-dir "${SPLITS_DIR}" \
      --predictions-root "${PREDICTIONS_ROOT}" \
      --fold "${FOLD}" \
      --backbone efficientnet_v2_s \
      --pooling "${POOLING}" \
      --top-k "${TOP_K}" \
      --min-probability "${MIN_PROBABILITY}" \
      --crop-size-mm "${CROP_SIZE_MM}" \
      --crop-image-size "${CROP_IMAGE_SIZE}" \
      --batch-size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --precision "${PRECISION}" \
      --accelerator "${ACCELERATOR}" \
      --devices "${DEVICES}" \
      --lr "${LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --num-workers "${NUM_WORKERS}" \
      --monitor val_mcc \
      --wandb-group "efficientnet_v2_s_detection_mil_${POOLING}" \
      "${EXTRA_ARGS[@]}"
  done
done

python3 -m src.luna16_detection_mil.aggregate \
  --output-dir "${OUTPUT_DIR}" \
  --poolings ${POOLINGS}
