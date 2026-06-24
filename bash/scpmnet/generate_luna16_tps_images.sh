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

PRED_ROOT="${PRED_ROOT:-outputs/scpmnet_luna16_10fold}"
SPLIT_DIR="${SPLIT_DIR:-/ssd2/domenico/datasets/LUNA16_preprocessed/cv_splits}"
DATA_ROOT="${DATA_ROOT:-/ssd2/domenico/datasets/LUNA16_preprocessed}"
OUTPUT_DIR_WAS_SET="${OUTPUT_DIR+x}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/scpmnet_luna16_10fold_tps_images}"
START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-9}"
TOP_K="${TOP_K:-5}"
SMOOTH="${SMOOTH:-5.0}"
ANCHOR_GRID_SIZE="${ANCHOR_GRID_SIZE:-10}"
SURFACE_CLIP_MARGIN="${SURFACE_CLIP_MARGIN:-40.0}"
FALLBACK_TOP_K="${FALLBACK_TOP_K:-3 1}"
BOUNDARY_MARGIN="${BOUNDARY_MARGIN:-8.0}"
MIN_PROBABILITY="${MIN_PROBABILITY:-}"
QC_GOOD_Z_RANGE="${QC_GOOD_Z_RANGE:-180.0}"
QC_GOOD_SLOPE_P99="${QC_GOOD_SLOPE_P99:-10.0}"
QC_GOOD_CURV_P99="${QC_GOOD_CURV_P99:-2.5}"
QC_MEDIUM_Z_RANGE="${QC_MEDIUM_Z_RANGE:-240.0}"
QC_MEDIUM_SLOPE_P99="${QC_MEDIUM_SLOPE_P99:-16.0}"
QC_MEDIUM_CURV_P99="${QC_MEDIUM_CURV_P99:-5.0}"
GOOD_WEIGHT="${GOOD_WEIGHT:-1.0}"
MEDIUM_WEIGHT="${MEDIUM_WEIGHT:-0.5}"
BAD_WEIGHT="${BAD_WEIGHT:-0.2}"
MAX_SCANS="${MAX_SCANS:-}"
TEST_RUN="${TEST_RUN:-false}"

EXTRA_ARGS=()
if [ "$TEST_RUN" = "true" ]; then
  START_FOLD="${TEST_FOLD:-0}"
  END_FOLD="${TEST_FOLD:-0}"
  MAX_SCANS="${MAX_SCANS:-2}"
  if [ -z "$OUTPUT_DIR_WAS_SET" ]; then
    OUTPUT_DIR="outputs/scpmnet_luna16_10fold_tps_images_test"
  fi
fi

if [ -n "$MAX_SCANS" ]; then
  EXTRA_ARGS+=(--max-scans "$MAX_SCANS")
fi

if [ -n "$MIN_PROBABILITY" ]; then
  EXTRA_ARGS+=(--min-probability "$MIN_PROBABILITY")
fi

FOLDS=()
for FOLD in $(seq "$START_FOLD" "$END_FOLD"); do
  FOLDS+=("$FOLD")
done

read -r -a FALLBACK_TOP_K_VALUES <<< "$FALLBACK_TOP_K"

python -m src.det.SCPMNet.generate_luna16_tps_images \
  --pred-root "$PRED_ROOT" \
  --split-dir "$SPLIT_DIR" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --top-k "$TOP_K" \
  --smooth "$SMOOTH" \
  --anchor-grid-size "$ANCHOR_GRID_SIZE" \
  --surface-clip-margin "$SURFACE_CLIP_MARGIN" \
  --fallback-top-k "${FALLBACK_TOP_K_VALUES[@]}" \
  --boundary-margin "$BOUNDARY_MARGIN" \
  --qc-good-z-range "$QC_GOOD_Z_RANGE" \
  --qc-good-slope-p99 "$QC_GOOD_SLOPE_P99" \
  --qc-good-curv-p99 "$QC_GOOD_CURV_P99" \
  --qc-medium-z-range "$QC_MEDIUM_Z_RANGE" \
  --qc-medium-slope-p99 "$QC_MEDIUM_SLOPE_P99" \
  --qc-medium-curv-p99 "$QC_MEDIUM_CURV_P99" \
  --good-weight "$GOOD_WEIGHT" \
  --medium-weight "$MEDIUM_WEIGHT" \
  --bad-weight "$BAD_WEIGHT" \
  "${EXTRA_ARGS[@]}" \
  --folds "${FOLDS[@]}" \
  --save-surfaces \
  --save-overlays \
  "$@"
