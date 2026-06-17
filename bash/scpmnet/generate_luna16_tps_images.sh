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
SPLIT_DIR="${SPLIT_DIR:-data/LUNA16_preprocessed/cv_splits}"
DATA_ROOT="${DATA_ROOT:-data/LUNA16_preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/scpmnet_luna16_10fold_tps_images}"
START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-9}"
TOP_K="${TOP_K:-5}"

FOLDS=()
for FOLD in $(seq "$START_FOLD" "$END_FOLD"); do
  FOLDS+=("$FOLD")
done

python -m src.det.SCPMNet.generate_luna16_tps_images \
  --pred-root "$PRED_ROOT" \
  --split-dir "$SPLIT_DIR" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --top-k "$TOP_K" \
  --folds "${FOLDS[@]}" \
  "$@"
