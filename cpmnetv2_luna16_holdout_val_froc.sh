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
SOURCE_CSV="${SOURCE_CSV:-$DATA_ROOT/cv_splits/luna16_fold0.csv}"
SPLIT_CSV="${SPLIT_CSV:-$DATA_ROOT/holdout_splits/luna16_holdout_seed233.csv}"
LABELS_CSV="${LABELS_CSV:-$SPLIT_CSV}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cpmnetv2_luna16_holdout}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cpmnetv2_luna16_holdout_seed233}"
SEED="${SEED:-233}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
DEVICES="${DEVICES:-[2]}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-3}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
VAL_FROC_START_EPOCH="${VAL_FROC_START_EPOCH:-80}"
VAL_FROC_BEFORE_START_EVERY_N_EPOCH="${VAL_FROC_BEFORE_START_EVERY_N_EPOCH:-100}"
CHECKPOINT_START_EPOCH="${CHECKPOINT_START_EPOCH:-120}"
MAX_EPOCHS="${MAX_EPOCHS:-170}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CROP_SIZE="${CROP_SIZE:-[96, 96, 96]}"
OVERLAP_SIZE="${OVERLAP_SIZE:-[24, 24, 24]}"
SPACING="${SPACING:-[1.0, 1.0, 1.0]}"
TOPK="${TOPK:-7}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "$SPLIT_CSV" ]; then
  python -m src.det.SCPMNet.prepare_luna16_holdout_split \
    --source-csv "$SOURCE_CSV" \
    --output-csv "$SPLIT_CSV" \
    --val-fraction "$VAL_FRACTION" \
    --test-fraction "$TEST_FRACTION" \
    --seed "$SEED"
fi

python -m src.det.CPMNetv2.train_lightning \
  --config-name train_lightning \
  dataset_name=luna16 \
  csv_path="$SPLIT_CSV" \
  images_dir="$DATA_ROOT" \
  annotations_dir="$DATA_ROOT" \
  labels_csv="$LABELS_CSV" \
  output_dir="$OUTPUT_DIR" \
  experiment_name="$EXPERIMENT_NAME" \
  seed="$SEED" \
  max_epochs="$MAX_EPOCHS" \
  batch_size="$BATCH_SIZE" \
  num_workers="$NUM_WORKERS" \
  crop_size="$CROP_SIZE" \
  overlap_size="$OVERLAP_SIZE" \
  spacing="$SPACING" \
  topk="$TOPK" \
  num_samples="$NUM_SAMPLES" \
  devices="$DEVICES" \
  use_wandb=True \
  wandb_name="$EXPERIMENT_NAME" \
  accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
  val_full_volume=true \
  check_val_every_n_epoch="$CHECK_VAL_EVERY_N_EPOCH" \
  val_froc_start_epoch="$VAL_FROC_START_EPOCH" \
  val_froc_before_start_every_n_epoch="$VAL_FROC_BEFORE_START_EVERY_N_EPOCH" \
  checkpoint_monitor=val/mean_froc \
  checkpoint_mode=max \
  checkpoint_filename='cpmnetv2_luna16_best' \
  checkpoint_every_n_epochs="$CHECK_VAL_EVERY_N_EPOCH" \
  checkpoint_start_epoch="$CHECKPOINT_START_EPOCH" \
  "$@"
