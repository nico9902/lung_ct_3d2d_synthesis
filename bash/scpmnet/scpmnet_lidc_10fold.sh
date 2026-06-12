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

DATA_ROOT="${DATA_ROOT:-/ssd2/domenico/datasets/lidc_process}"
SPLIT_DIR="${SPLIT_DIR:-$DATA_ROOT/cv_splits}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/scpmnet_lidc_10fold}"
DEVICES="${DEVICES:-[0]}"
BATCH_SIZE="${BATCH_SIZE:-24}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-9}"
SEED="${SEED:-233}"

MAX_EPOCHS="${MAX_EPOCHS:-170}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
VAL_FROC_START_EPOCH="${VAL_FROC_START_EPOCH:-80}"
VAL_FROC_BEFORE_START_EVERY_N_EPOCH="${VAL_FROC_BEFORE_START_EVERY_N_EPOCH:-10}"
CHECKPOINT_START_EPOCH="${CHECKPOINT_START_EPOCH:-0}"

for FOLD in $(seq "$START_FOLD" "$END_FOLD"); do
  CSV_PATH="$SPLIT_DIR/lidc_fold${FOLD}.csv"
  EXPERIMENT_NAME="scpmnet_paper_lidc_fold${FOLD}"

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
    val_full_volume=false \
    val_modes='[random_crop_loss]' \
    val_fixed_crop_seed="$SEED" \
    val_random_crop_samples_per_volume=4 \
    test_full_volume=true \
    evaluate_froc=true \
    max_epochs="$MAX_EPOCHS" \
    evaluate_val_froc=false \
    evaluate_test_froc=true \
    check_val_every_n_epoch="$CHECK_VAL_EVERY_N_EPOCH" \
    val_froc_start_epoch=null \
    val_froc_before_start_every_n_epoch=null \
    checkpoint_monitor=val/random_crop/loss \
    checkpoint_mode=min \
    checkpoint_filename="'epoch\={epoch:03d}-val_random_crop/loss\={val/random_crop/loss:.4f}'" \
    checkpoint_every_n_epochs="$CHECK_VAL_EVERY_N_EPOCH" \
    checkpoint_start_epoch="$CHECKPOINT_START_EPOCH" \
    "$@"
done
