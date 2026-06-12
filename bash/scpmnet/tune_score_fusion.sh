#!/bin/bash

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR" || exit 1

VENV_PATH="${VENV_PATH:-myenv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

CSV="${CSV:-/ssd2/domenico/datasets/lidc_process/lidc_labels_clean_flagged_smaller.csv}"
ROOT="${ROOT:-/ssd2/domenico/datasets/lidc_process}"
RESCORED="${RESCORED:-outputs/scpmnet/scpmnet_paper/fp_reduction/rescored_test/test_predictions_rescored.csv}"
OUT="${OUT:-outputs/scpmnet/scpmnet_paper/fp_reduction/fusion_tuning}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
DEVICE="${DEVICE:-cuda:0}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-bf16-mixed}"

python src/det/SCPMNet/tune_score_fusion.py \
  --rescored-candidates "$RESCORED" \
  --csv-path "$CSV" \
  --data-root "$ROOT" \
  --split test \
  --output-dir "$OUT" \
  --scpm-exponents "${SCPM_EXPONENTS:-0,0.25,0.5,0.75,1,1.25,1.5,2}" \
  --classifier-exponents "${CLASSIFIER_EXPONENTS:-0,0.25,0.5,0.75,1,1.25,1.5,2,3}"
