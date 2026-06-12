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

DATA_ROOT="${DATA_ROOT:-/ssd2/domenico/datasets/LUNA16_preprocessed}"
ANNOTATIONS_CSV="${ANNOTATIONS_CSV:-/ssd2/domenico/datasets/LUNA16/annotations/annotations.csv}"
SPLIT_DIR="${SPLIT_DIR:-$DATA_ROOT/cv_splits}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/scpmnet_luna16_10fold}"
DEVICES="${DEVICES:-[3]}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-3}"
START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-9}"

# python -m src.det.SCPMNet.prepare_luna16_cv_splits \
#   --preprocessed-root "$DATA_ROOT" \
#   --annotations-csv "$ANNOTATIONS_CSV" \
#   --output-dir "$SPLIT_DIR"

for FOLD in $(seq "$START_FOLD" "$END_FOLD"); do
  CSV_PATH="$SPLIT_DIR/luna16_fold${FOLD}.csv"
  EXPERIMENT_NAME="scpmnet_paper_luna16_fold${FOLD}"

  python -m src.det.SCPMNet.train_lightning \
    --config-name train_lightning_paper \
    csv_path="$CSV_PATH" \
    data_root="$DATA_ROOT" \
    output_dir="$OUTPUT_DIR" \
    experiment_name="$EXPERIMENT_NAME" \
    batch_size="$BATCH_SIZE" \
    devices="$DEVICES" \
    use_wandb=True \
    wandb_name="$EXPERIMENT_NAME" \
    samples_per_volume=1 \
    accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
    "$@"
done
