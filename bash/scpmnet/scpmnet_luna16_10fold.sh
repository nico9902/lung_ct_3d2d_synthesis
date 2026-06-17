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
DEVICES="${DEVICES:-[0]}"
SEED="${SEED:-233}"
BATCH_SIZE="${BATCH_SIZE:-24}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-170}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
CHECKPOINT_START_EPOCH="${CHECKPOINT_START_EPOCH:-0}"
CHECKPOINT="${CHECKPOINT:-null}"
START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-9}"
USE_WANDB="${USE_WANDB:-True}"

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
    seed="$SEED" \
    output_dir="$OUTPUT_DIR" \
    experiment_name="$EXPERIMENT_NAME" \
    batch_size="$BATCH_SIZE" \
    use_wandb="$USE_WANDB" \
    wandb_name="$EXPERIMENT_NAME" \
    devices="$DEVICES" \
    use_wandb=True \
    wandb_name="$EXPERIMENT_NAME" \
    samples_per_volume=1 \
    accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
    checkpoint="$CHECKPOINT" \
    test_only=false \
    val_full_volume=false \
    val_modes='[random_crop_loss]' \
    val_fixed_crop_seed="$SEED" \
    val_random_crop_samples_per_volume=4 \
    test_full_volume=true \
    evaluate_froc=true \
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
