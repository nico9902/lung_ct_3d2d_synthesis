#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"

VENV_PATH="${VENV_PATH:-myenv}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
  source "${VENV_PATH}/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

LUNA16_PROCESSED_DIR="${LUNA16_PROCESSED_DIR:-/ssd2/domenico/datasets/LUNA16_preprocessed}"
SPLITS_DIR="${SPLITS_DIR:-${LUNA16_PROCESSED_DIR}/cv_splits}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
SPLIT="${SPLIT:-test}"
CSV_FILE_PREFIX="${CSV_FILE_PREFIX:-${SPLITS_DIR}/luna16_classification_fold}"
CSV_FILE_SUFFIX="${CSV_FILE_SUFFIX:-.csv}"
PRED_ROOT="${PRED_ROOT:-outputs/scpmnet_luna16_10fold}"
PREDICTION_NAME="${PREDICTION_NAME:-test_predictions.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/luna16_saliency_synthetic_detector_top5}"
TOP_K="${TOP_K:-5}"
NUM_CONTOUR_POINTS="${NUM_CONTOUR_POINTS:-4}"
MIN_PROBABILITY="${MIN_PROBABILITY:-}"
RBF_SMOOTH="${RBF_SMOOTH:-0.1}"
NUM_BOUNDARY_ANCHORS="${NUM_BOUNDARY_ANCHORS:-24}"
USE_LUNG_VOLUME_ANCHORS="${USE_LUNG_VOLUME_ANCHORS:-true}"
LUNG_ANCHOR_ERODE_ITERATIONS="${LUNG_ANCHOR_ERODE_ITERATIONS:-1}"
SNAP_SURFACE_TO_LUNG="${SNAP_SURFACE_TO_LUNG:-false}"
ANCHOR_MIN_LUNG_AREA_FRACTION="${ANCHOR_MIN_LUNG_AREA_FRACTION:-0.35}"
USE_SAVED_LUNG_MASKS="${USE_SAVED_LUNG_MASKS:-true}"
LUNG_MASK_METHOD="${LUNG_MASK_METHOD:-body_threshold}"
NORMALIZED_AIR_THRESHOLD="${NORMALIZED_AIR_THRESHOLD:-0.35}"
HU_AIR_MIN="${HU_AIR_MIN:--1000}"
HU_AIR_MAX="${HU_AIR_MAX:--320}"
BODY_THRESHOLD_PERCENTILE="${BODY_THRESHOLD_PERCENTILE:-1}"
LUNG_COMPONENT_COUNT="${LUNG_COMPONENT_COUNT:-2}"
LUNGMASK_MODEL_NAME="${LUNGMASK_MODEL_NAME:-R231}"
LUNGMASK_FORCE_CPU="${LUNGMASK_FORCE_CPU:-false}"
LUNG_WINDOW_CENTER="${LUNG_WINDOW_CENTER:--600}"
LUNG_WINDOW_WIDTH="${LUNG_WINDOW_WIDTH:-1500}"
FALLBACK_MID_SLICE="${FALLBACK_MID_SLICE:-false}"
REPORT_LUNG_COVERAGE="${REPORT_LUNG_COVERAGE:-true}"
SAVE_SURFACE_GRID="${SAVE_SURFACE_GRID:-true}"

export MPLBACKEND="${MPLBACKEND:-Agg}"

for FOLD in ${FOLDS}; do
  CSV_FILE="${CSV_FILE_PREFIX}${FOLD}${CSV_FILE_SUFFIX}"

  if [[ "${CSV_FILE}" == *"{"* || "${CSV_FILE}" == *"}"* ]]; then
    echo "Error: CSV path still contains a brace placeholder: ${CSV_FILE}" >&2
    echo "Set CSV_FILE_PREFIX and CSV_FILE_SUFFIX instead of using a {fold} template." >&2
    exit 1
  fi

  EXTRA_ARGS=()
  if [ -n "${MIN_PROBABILITY}" ]; then
    EXTRA_ARGS+=(--min-probability "${MIN_PROBABILITY}")
  fi
  if [ "${FALLBACK_MID_SLICE}" = "true" ]; then
    EXTRA_ARGS+=(--fallback-mid-slice)
  fi
  if [ "${REPORT_LUNG_COVERAGE}" = "true" ]; then
    EXTRA_ARGS+=(--report-lung-coverage)
  fi
  if [ "${SAVE_SURFACE_GRID}" = "true" ]; then
    EXTRA_ARGS+=(--save-surface-grid)
  fi
  if [ "${USE_LUNG_VOLUME_ANCHORS}" = "true" ]; then
    EXTRA_ARGS+=(--use-lung-volume-anchors)
  else
    EXTRA_ARGS+=(--no-use-lung-volume-anchors)
  fi
  if [ "${SNAP_SURFACE_TO_LUNG}" = "true" ]; then
    EXTRA_ARGS+=(--snap-surface-to-lung)
  else
    EXTRA_ARGS+=(--no-snap-surface-to-lung)
  fi
  if [ "${USE_SAVED_LUNG_MASKS}" = "true" ]; then
    EXTRA_ARGS+=(--use-saved-lung-masks)
  else
    EXTRA_ARGS+=(--no-use-saved-lung-masks)
  fi
  if [ "${LUNGMASK_FORCE_CPU}" = "true" ]; then
    EXTRA_ARGS+=(--lungmask-force-cpu)
  else
    EXTRA_ARGS+=(--no-lungmask-force-cpu)
  fi

  echo "Generating detector-driven LUNA16 saliency surfaces for fold ${FOLD}"
  echo "  CSV: ${CSV_FILE}"
  echo "  Detector root: ${PRED_ROOT}"
  echo "  Split: ${SPLIT}"
  echo "  Top-k: ${TOP_K}"
  echo "  Control points per detection: $((1 + NUM_CONTOUR_POINTS))"
  echo "  Output: ${OUTPUT_ROOT}"

  python3 -m src.luna16_synthetic_2d.detector_saliency \
    --fold "${FOLD}" \
    --split "${SPLIT}" \
    --csv-file "${CSV_FILE}" \
    --processed-dir "${LUNA16_PROCESSED_DIR}" \
    --pred-root "${PRED_ROOT}" \
    --prediction-name "${PREDICTION_NAME}" \
    --save-path "${OUTPUT_ROOT}" \
    --top-k "${TOP_K}" \
    --num-contour-points "${NUM_CONTOUR_POINTS}" \
    --rbf-smooth "${RBF_SMOOTH}" \
    --num-boundary-anchors "${NUM_BOUNDARY_ANCHORS}" \
    --lung-anchor-erode-iterations "${LUNG_ANCHOR_ERODE_ITERATIONS}" \
    --anchor-min-lung-area-fraction "${ANCHOR_MIN_LUNG_AREA_FRACTION}" \
    --lung-mask-method "${LUNG_MASK_METHOD}" \
    --normalized-air-threshold "${NORMALIZED_AIR_THRESHOLD}" \
    --hu-air-min "${HU_AIR_MIN}" \
    --hu-air-max "${HU_AIR_MAX}" \
    --body-threshold-percentile "${BODY_THRESHOLD_PERCENTILE}" \
    --lung-component-count "${LUNG_COMPONENT_COUNT}" \
    --lungmask-model-name "${LUNGMASK_MODEL_NAME}" \
    --lung-window-center "${LUNG_WINDOW_CENTER}" \
    --lung-window-width "${LUNG_WINDOW_WIDTH}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
done
