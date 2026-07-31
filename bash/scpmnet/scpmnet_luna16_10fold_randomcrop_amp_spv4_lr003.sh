#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
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
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/scpmnet_luna16_10fold_normauto_amp_spv4_lr003_randomcrop}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-scpmnet_paper_luna16_normauto_amp_spv4_lr003_fold}"
NORMALIZED_VOLUME_CACHE_DIR="${NORMALIZED_VOLUME_CACHE_DIR:-outputs/scpmnet_luna16_10fold_fpr_top100_focal_balanced_average_normauto/normalized_volume_cache_auto}"

START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-9}"
SEED="${SEED:-233}"

# On the server, with PCI_BUS_ID ordering, the 80GB A100 has been the physical GPU 3.
# Override CUDA_VISIBLE_DEVICES at launch if the free GPU changes.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PRECISION="${PRECISION:-16-mixed}"
LR="${LR:-0.003}"
BATCH_SIZE="${BATCH_SIZE:-24}"
SAMPLES_PER_VOLUME="${SAMPLES_PER_VOLUME:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-170}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
CHECKPOINT_START_EPOCH="${CHECKPOINT_START_EPOCH:-0}"
CHECKPOINT="${CHECKPOINT:-null}"
VAL_RANDOM_CROP_SAMPLES_PER_VOLUME="${VAL_RANDOM_CROP_SAMPLES_PER_VOLUME:-4}"

MONITOR_FINITE_VALUES="${MONITOR_FINITE_VALUES:-true}"
FINITE_MONITOR_CHECK_LOGGED_METRICS="${FINITE_MONITOR_CHECK_LOGGED_METRICS:-true}"
FINITE_MONITOR_CHECK_GRADIENTS="${FINITE_MONITOR_CHECK_GRADIENTS:-true}"
FINITE_MONITOR_EVERY_N_TRAIN_STEPS="${FINITE_MONITOR_EVERY_N_TRAIN_STEPS:-1}"
DETECT_ANOMALY="${DETECT_ANOMALY:-false}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-0.0}"
GRADIENT_CLIP_ALGORITHM="${GRADIENT_CLIP_ALGORITHM:-norm}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-60}"

WANDB_PROJECT="${WANDB_PROJECT:-lung_ct_3d2d_synthesis_detection}"
WANDB_MODE="${WANDB_MODE:-online}"
USE_WANDB="${USE_WANDB:-true}"

RUN_AGGREGATE="${RUN_AGGREGATE:-true}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/scpmnet_luna16_amp_spv4_lr003_randomcrop_${RUN_TAG}}"
mkdir -p "$LOG_DIR"

echo "==== SCPMNet LUNA16 random crop loss AMP run"
echo "==== project: $PROJECT_DIR"
echo "==== output: $OUTPUT_ROOT"
echo "==== folds: $START_FOLD..$END_FOLD"
echo "==== CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES | devices: $DEVICES | precision: $PRECISION"
echo "==== batch_size: $BATCH_SIZE | samples_per_volume: $SAMPLES_PER_VOLUME | effective batch: $((BATCH_SIZE * SAMPLES_PER_VOLUME))"
echo "==== lr: $LR | finite monitor: $MONITOR_FINITE_VALUES | gradient check interval: $FINITE_MONITOR_EVERY_N_TRAIN_STEPS"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L | tee "$LOG_DIR/gpu_list.txt"
  (
    while true; do
      date
      nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv
      sleep "$GPU_MONITOR_INTERVAL"
    done
  ) >> "$LOG_DIR/gpu_monitor.csv" 2>&1 &
  GPU_MONITOR_PID=$!
  trap 'kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true' EXIT
fi

for FOLD in $(seq "$START_FOLD" "$END_FOLD"); do
  CSV_PATH="$SPLIT_DIR/luna16_fold${FOLD}.csv"
  EXPERIMENT_NAME="${EXPERIMENT_PREFIX}${FOLD}"
  WANDB_NAME="${WANDB_NAME_PREFIX:-scpmnet_luna16_amp_spv4_lr003_randomcrop_fold}${FOLD}"
  LOG_FILE="$LOG_DIR/fold${FOLD}.log"

  echo "==== Fold $FOLD | csv: $CSV_PATH"
  echo "==== Fold $FOLD | output: $OUTPUT_ROOT/$EXPERIMENT_NAME"
  echo "==== Fold $FOLD | log: $LOG_FILE"

  stdbuf -oL -eL python -m src.det.SCPMNet.train_lightning \
    --config-name train_lightning_paper \
    csv_path="$CSV_PATH" \
    data_root="$DATA_ROOT" \
    seed="$SEED" \
    output_dir="$OUTPUT_ROOT" \
    experiment_name="$EXPERIMENT_NAME" \
    batch_size="$BATCH_SIZE" \
    num_workers="$NUM_WORKERS" \
    max_epochs="$MAX_EPOCHS" \
    accelerator="$ACCELERATOR" \
    devices="$DEVICES" \
    precision="$PRECISION" \
    lr="$LR" \
    intensity_mode=auto \
    normalized_volume_cache_dir="$NORMALIZED_VOLUME_CACHE_DIR" \
    samples_per_volume="$SAMPLES_PER_VOLUME" \
    accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
    checkpoint="$CHECKPOINT" \
    test_only=false \
    val_full_volume=false \
    val_modes='[random_crop_loss]' \
    val_fixed_crop_seed="$SEED" \
    val_random_crop_samples_per_volume="$VAL_RANDOM_CROP_SAMPLES_PER_VOLUME" \
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
    monitor_finite_values="$MONITOR_FINITE_VALUES" \
    finite_monitor_check_logged_metrics="$FINITE_MONITOR_CHECK_LOGGED_METRICS" \
    finite_monitor_check_gradients="$FINITE_MONITOR_CHECK_GRADIENTS" \
    finite_monitor_every_n_train_steps="$FINITE_MONITOR_EVERY_N_TRAIN_STEPS" \
    detect_anomaly="$DETECT_ANOMALY" \
    gradient_clip_val="$GRADIENT_CLIP_VAL" \
    gradient_clip_algorithm="$GRADIENT_CLIP_ALGORITHM" \
    use_wandb="$USE_WANDB" \
    wandb_project="$WANDB_PROJECT" \
    wandb_mode="$WANDB_MODE" \
    wandb_name="$WANDB_NAME" \
    "$@" 2>&1 | tee "$LOG_FILE"
done

if [ "$RUN_AGGREGATE" = "true" ]; then
  python -m src.det.SCPMNet.aggregate_luna16_cv \
    --output-root "$OUTPUT_ROOT" \
    --split-dir "$SPLIT_DIR" \
    --prediction-name test_predictions.csv \
    --froc-name test_froc.csv \
    --out-dir "$OUTPUT_ROOT/cv_aggregate" 2>&1 | tee "$LOG_DIR/aggregate.log"
fi

echo "SCPMNet LUNA16 AMP spv4 lr003 random-crop run completed."
