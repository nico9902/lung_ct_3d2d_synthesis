#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

VENV_PATH="${VENV_PATH:-myenv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

DATA_ROOT="${DATA_ROOT:-/ssd2/domenico/datasets/LUNA16_preprocessed}"
SPLIT_DIR="${SPLIT_DIR:-$DATA_ROOT/cv_splits}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/scpmnet_luna16_10fold_normauto}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-scpmnet_paper_luna16_normauto_fold}"
NORMALIZED_VOLUME_CACHE_DIR="${NORMALIZED_VOLUME_CACHE_DIR:-outputs/scpmnet_luna16_10fold_fpr_top100_focal_balanced_average_normauto/normalized_volume_cache_auto}"

START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-9}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32}"

BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-170}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
CHECKPOINT_START_EPOCH="${CHECKPOINT_START_EPOCH:-0}"
CHECKPOINT="${CHECKPOINT:-null}"
WANDB_PROJECT="${WANDB_PROJECT:-lung_ct_3d2d_synthesis_detection}"
WANDB_MODE="${WANDB_MODE:-online}"
USE_WANDB="${USE_WANDB:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for FOLD in $(seq "$START_FOLD" "$END_FOLD"); do
  CSV_PATH="$SPLIT_DIR/luna16_fold${FOLD}.csv"
  EXPERIMENT_NAME="${EXPERIMENT_PREFIX}${FOLD}"
  WANDB_NAME="${WANDB_NAME_PREFIX:-scpmnet_luna16_normauto_a100_fold}${FOLD}"

  echo "==== Detector fold $FOLD | csv: $CSV_PATH"
  echo "==== Detector fold $FOLD | output: $OUTPUT_ROOT/$EXPERIMENT_NAME"
  echo "==== Detector fold $FOLD | cache: $NORMALIZED_VOLUME_CACHE_DIR"

  python -m src.det.SCPMNet.train_lightning \
    --config-name train_lightning_paper \
    csv_path="$CSV_PATH" \
    data_root="$DATA_ROOT" \
    output_dir="$OUTPUT_ROOT" \
    experiment_name="$EXPERIMENT_NAME" \
    batch_size="$BATCH_SIZE" \
    num_workers="$NUM_WORKERS" \
    max_epochs="$MAX_EPOCHS" \
    accelerator="$ACCELERATOR" \
    devices="$DEVICES" \
    precision="$PRECISION" \
    intensity_mode=auto \
    normalized_volume_cache_dir="$NORMALIZED_VOLUME_CACHE_DIR" \
    samples_per_volume=1 \
    accumulate_grad_batches=1 \
    checkpoint="$CHECKPOINT" \
    test_only=false \
    val_full_volume=false \
    val_modes='[random_crop_loss]' \
    val_fixed_crop_seed=233 \
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
    use_wandb="$USE_WANDB" \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="$WANDB_NAME"
done

python -m src.det.SCPMNet.aggregate_luna16_cv \
  --output-root "$OUTPUT_ROOT" \
  --split-dir "$SPLIT_DIR" \
  --prediction-name test_predictions.csv \
  --froc-name test_froc.csv \
  --out-dir "$OUTPUT_ROOT/cv_aggregate"

echo "LUNA16 detector normauto completed."
