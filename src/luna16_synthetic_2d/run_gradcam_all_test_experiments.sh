#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_DIR}"

VENV_PATH="${VENV_PATH:-myenv}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
  source "${VENV_PATH}/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

SELECTION_CSV="${SELECTION_CSV:-docs/luna16_synthetic_2d_class_rankings/luna16_synthetic_2d_class_rankings.csv}"
TARGET_CLASS="${TARGET_CLASS:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/luna16_synthetic_2d_gradcam_all_${TARGET_CLASS}}"
TOP_N="${TOP_N:-999999}"
DEVICE="${DEVICE:-cuda:2}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_DIR}/.matplotlib_cache}"
SYNTHETIC_ROOT_BASE="${SYNTHETIC_ROOT_BASE:-/ssd2/domenico/datasets/synthetic_2d}"
BACKBONES="${BACKBONES:-efficientnet_v2_s}"
EXPERIMENTS="${EXPERIMENTS:-luna16_synthetic_2d_gt luna16_synthetic_2d_top5_minprob0.5 luna16_synthetic_2d_top7_minprob0.3_rbf luna16_synthetic_2d_top7_minprob0.3_shepard luna16_synthetic_2d_top4_minprob0.5_rbf luna16_synthetic_2d_top4_minprob0.5_shepard luna16_synthetic_2d_top3_minprob0.5_rbf luna16_synthetic_2d_top3_minprob0.5_shepard}"
LABELS="${LABELS:-benign malignant}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
FAIL_MISSING_CHECKPOINT="${FAIL_MISSING_CHECKPOINT:-true}"
REPORT_MAX_ROWS="${REPORT_MAX_ROWS:-1000}"

mkdir -p "${MPLCONFIGDIR}"
export MPLCONFIGDIR

EXTRA_ARGS=()
if [ "${SKIP_EXISTING}" = "true" ]; then
  EXTRA_ARGS+=(--skip-existing)
fi
if [ "${FAIL_MISSING_CHECKPOINT}" = "true" ]; then
  EXTRA_ARGS+=(--fail-missing-checkpoint)
fi

echo "Exporting all-test LUNA16 synthetic 2D GradCAM"
echo "  Selection CSV: ${SELECTION_CSV}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Experiments: ${EXPERIMENTS}"
echo "  Backbones: ${BACKBONES}"
echo "  Labels: ${LABELS}"
echo "  Target class: ${TARGET_CLASS}"
echo "  Device: ${DEVICE}"
echo "  Synthetic root base: ${SYNTHETIC_ROOT_BASE}"

python3 -m src.luna16_synthetic_2d.export_gradcam \
  --selection-csv "${SELECTION_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --top-n "${TOP_N}" \
  --target-class "${TARGET_CLASS}" \
  --device "${DEVICE}" \
  --synthetic-root-base "${SYNTHETIC_ROOT_BASE}" \
  --report-max-rows "${REPORT_MAX_ROWS}" \
  --experiments ${EXPERIMENTS} \
  --backbones ${BACKBONES} \
  --rank-kinds all \
  --labels ${LABELS} \
  "${EXTRA_ARGS[@]}" \
  "$@"
