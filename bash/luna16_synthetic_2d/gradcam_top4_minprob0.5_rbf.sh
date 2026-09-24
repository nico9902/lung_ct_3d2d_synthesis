#!/usr/bin/env bash
# Export Grad-CAM explanations of the proposed method (adaptive RBF, top-4,
# p >= 0.5, EfficientNetV2-S) for the test split of every fold.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-luna16_synthetic_2d_cpmnetv2_bf16_top4_minprob0.50_rbf_v100}"
BACKBONE="${BACKBONE:-efficientnet_v2_s}"
SPLIT_NAME="${SPLIT_NAME:-test}"
SYNTHETIC_IMAGES_DIR="${SYNTHETIC_IMAGES_DIR:-${SYNTHETIC_ROOT}/luna16_saliency_synthetic_detector_cpmnetv2_bf16_top4_minprob0.50_rbf}"
RUN_ROOT="${RUN_ROOT:-outputs/${EXPERIMENT_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/luna16_synthetic_2d_gradcam/${EXPERIMENT_NAME}_${BACKBONE}}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-*.ckpt}"
DEVICE="${DEVICE:-auto}"
TARGET_CLASS="${TARGET_CLASS:-predicted}"
SELECTION="${SELECTION:-balanced_highest_score}"
MAX_SAMPLES="${MAX_SAMPLES:-24}"
IMAGE_SIZE="${IMAGE_SIZE:-256 384}"
IMAGE_SUFFIX="${IMAGE_SUFFIX:-_tps_top5.npy}"
EXCLUDE_SAMPLE_IDS="${EXCLUDE_SAMPLE_IDS:-120188208010880}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_DIR}/.matplotlib_cache}"

mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_ROOT}"
export MPLCONFIGDIR

echo "Exporting Grad-CAM for proposed method"
echo "  Method: detector-guided RBF, top-k=4, min_probability=0.5"
echo "  Experiment: ${EXPERIMENT_NAME}"
echo "  Backbone: ${BACKBONE}"
echo "  Folds: ${FOLDS}"
echo "  Synthetic images: ${SYNTHETIC_IMAGES_DIR}"
echo "  Processed data: ${DATA_ROOT}"
echo "  Excluded samples: ${EXCLUDE_SAMPLE_IDS}"
echo "  Output root: ${OUTPUT_ROOT}"
echo "  Device: ${DEVICE}"

shopt -s nullglob

for FOLD in ${FOLDS}; do
  RUN_DIR="${RUN_ROOT}/fold_${FOLD}/${BACKBONE}"
  CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
  SPLIT_CSV="${SPLITS_DIR}/luna16_classification_fold${FOLD}.csv"
  FOLD_OUTPUT_DIR="${OUTPUT_ROOT}/fold_${FOLD}"

  CHECKPOINTS=("${CHECKPOINT_DIR}"/${CHECKPOINT_GLOB})
  FILTERED_CHECKPOINTS=()
  for CANDIDATE in "${CHECKPOINTS[@]}"; do
    if [[ "$(basename "${CANDIDATE}")" != last* ]]; then
      FILTERED_CHECKPOINTS+=("${CANDIDATE}")
    fi
  done

  if [ "${#FILTERED_CHECKPOINTS[@]}" -eq 0 ]; then
    echo "Missing checkpoint for fold ${FOLD}: ${CHECKPOINT_DIR}/${CHECKPOINT_GLOB}" >&2
    exit 1
  fi

  CHECKPOINT="${FILTERED_CHECKPOINTS[0]}"
  echo "Fold ${FOLD}: ${CHECKPOINT}"

  python -m src.luna16_synthetic_2d.explain_gradcam \
    --checkpoint "${CHECKPOINT}" \
    --run-dir "${RUN_DIR}" \
    --synthetic-images-dir "${SYNTHETIC_IMAGES_DIR}" \
    --split-csv "${SPLIT_CSV}" \
    --fold "${FOLD}" \
    --split "${SPLIT_NAME}" \
    --backbone "${BACKBONE}" \
    --processed-dir "${DATA_ROOT}" \
    --gt-overlay auto \
    --clip-gt-to-lung \
    --image-size ${IMAGE_SIZE} \
    --image-suffix "${IMAGE_SUFFIX}" \
    --exclude-sample-ids ${EXCLUDE_SAMPLE_IDS} \
    --selection "${SELECTION}" \
    --target-class "${TARGET_CLASS}" \
    --require-malignant-gt-hit \
    --require-gt-inside-visible-lung \
    --figure-layout paper_overlay \
    --max-samples "${MAX_SAMPLES}" \
    --device "${DEVICE}" \
    --output-dir "${FOLD_OUTPUT_DIR}" \
    "$@"
done

echo "Done. Grad-CAM outputs written under ${OUTPUT_ROOT}"
