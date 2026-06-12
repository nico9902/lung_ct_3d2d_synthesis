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
SPLIT_DIR="${SPLIT_DIR:-$DATA_ROOT/cv_splits}"
FOLD="${FOLD:-0}"
CSV_PATH="${CSV_PATH:-$SPLIT_DIR/luna16_fold${FOLD}.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/scpmnet_luna16_fold${FOLD}_val_loss_modes}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-scpmnet_paper_luna16_fold${FOLD}_val_loss_full_fixed_random_test_froc}"
SEED="${SEED:-233}"
DEVICES="${DEVICES:-[0]}"
BATCH_SIZE="${BATCH_SIZE:-24}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
SAMPLES_PER_VOLUME="${SAMPLES_PER_VOLUME:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-170}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
CHECKPOINT_START_EPOCH="${CHECKPOINT_START_EPOCH:-0}"
CHECKPOINT="${CHECKPOINT:-null}"
USE_WANDB="${USE_WANDB:-True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m src.det.SCPMNet.train_lightning \
  --config-name train_lightning_paper \
  csv_path="$CSV_PATH" \
  data_root="$DATA_ROOT" \
  output_dir="$OUTPUT_DIR" \
  experiment_name="$EXPERIMENT_NAME" \
  seed="$SEED" \
  max_epochs="$MAX_EPOCHS" \
  batch_size="$BATCH_SIZE" \
  num_workers="$NUM_WORKERS" \
  devices="$DEVICES" \
  use_wandb="$USE_WANDB" \
  wandb_name="$EXPERIMENT_NAME" \
  samples_per_volume="$SAMPLES_PER_VOLUME" \
  accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
  checkpoint="$CHECKPOINT" \
  test_only=false \
  val_full_volume=false \
  val_modes='[full_volume_loss,fixed_crop_loss,random_crop_loss]' \
  val_fixed_crop_seed="$SEED" \
  val_random_crop_samples_per_volume=4 \
  test_full_volume=true \
  evaluate_froc=true \
  evaluate_val_froc=false \
  evaluate_test_froc=true \
  check_val_every_n_epoch="$CHECK_VAL_EVERY_N_EPOCH" \
  val_froc_start_epoch=null \
  val_froc_before_start_every_n_epoch=null \
  checkpoint_monitor=val/full_volume/loss \
  checkpoint_mode=min \
  checkpoint_filename="'epoch\={epoch:03d}-val_full_volume_loss\={val/full_volume/loss:.4f}'" \
  checkpoint_every_n_epochs="$CHECK_VAL_EVERY_N_EPOCH" \
  checkpoint_start_epoch="$CHECKPOINT_START_EPOCH" \
  "$@"
