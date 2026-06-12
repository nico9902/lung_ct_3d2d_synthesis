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
SOURCE_CSV="${SOURCE_CSV:-$DATA_ROOT/cv_splits/luna16_fold0.csv}"
SPLIT_CSV="${SPLIT_CSV:-$DATA_ROOT/holdout_splits/luna16_holdout_seed233.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/scpmnet_luna16_holdout}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-scpmnet_paper_luna16_holdout_seed233}"
SEED="${SEED:-233}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
DEVICES="${DEVICES:-[0]}"
BATCH_SIZE="${BATCH_SIZE:-24}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
VAL_FROC_START_EPOCH="${VAL_FROC_START_EPOCH:-80}"
VAL_FROC_BEFORE_START_EVERY_N_EPOCH="${VAL_FROC_BEFORE_START_EVERY_N_EPOCH:-10}"
CHECKPOINT_START_EPOCH="${CHECKPOINT_START_EPOCH:-100}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "$SPLIT_CSV" ]; then
  if [ -f "$SOURCE_CSV" ]; then
    python -m src.det.SCPMNet.prepare_luna16_holdout_split \
      --source-csv "$SOURCE_CSV" \
      --output-csv "$SPLIT_CSV" \
      --val-fraction "$VAL_FRACTION" \
      --test-fraction "$TEST_FRACTION" \
      --seed "$SEED"
  else
    python -m src.det.SCPMNet.prepare_luna16_holdout_split \
      --preprocessed-root "$DATA_ROOT" \
      --annotations-csv "$ANNOTATIONS_CSV" \
      --output-csv "$SPLIT_CSV" \
      --val-fraction "$VAL_FRACTION" \
      --test-fraction "$TEST_FRACTION" \
      --seed "$SEED"
  fi
fi

python -m src.det.SCPMNet.train_lightning \
  --config-name train_lightning_paper \
  csv_path="$SPLIT_CSV" \
  data_root="$DATA_ROOT" \
  output_dir="$OUTPUT_DIR" \
  experiment_name="$EXPERIMENT_NAME" \
  seed="$SEED" \
  batch_size="$BATCH_SIZE" \
  devices="$DEVICES" \
  use_wandb=True \
  wandb_name="$EXPERIMENT_NAME" \
  samples_per_volume=1 \
  accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
  val_full_volume=true \
  check_val_every_n_epoch="$CHECK_VAL_EVERY_N_EPOCH" \
  val_froc_start_epoch="$VAL_FROC_START_EPOCH" \
  val_froc_before_start_every_n_epoch="$VAL_FROC_BEFORE_START_EVERY_N_EPOCH" \
  checkpoint_monitor=val/mean_froc \
  checkpoint_mode=max \
  checkpoint_filename='epoch={epoch:03d}-val_mean_froc={val_mean_froc:.4f}' \
  checkpoint_every_n_epochs="$CHECK_VAL_EVERY_N_EPOCH" \
  checkpoint_start_epoch="$CHECKPOINT_START_EPOCH" \
  "$@"
