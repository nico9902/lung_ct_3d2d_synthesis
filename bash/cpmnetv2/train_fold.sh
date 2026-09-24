#!/usr/bin/env bash
# Train CPMNetv2 on a single LUNA16 split CSV, selecting the checkpoint by
# validation FROC. Extra arguments are forwarded as Hydra overrides.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SPLIT_CSV="${SPLIT_CSV:-${SPLITS_DIR}/luna16_fold0.csv}"
LABELS_CSV="${LABELS_CSV:-${SPLIT_CSV}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cpmnetv2_luna16}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cpmnetv2_luna16_fold0}"
SEED="${SEED:-233}"
DEVICES="${DEVICES:-[0]}"
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

if [[ ! -f "${SPLIT_CSV}" ]]; then
  echo "Missing split CSV: ${SPLIT_CSV}" >&2
  exit 1
fi

python -m src.det.CPMNetv2.train_lightning \
  --config-name train_lightning \
  dataset_name=luna16 \
  csv_path="${SPLIT_CSV}" \
  images_dir="${DATA_ROOT}" \
  annotations_dir="${DATA_ROOT}" \
  labels_csv="${LABELS_CSV}" \
  output_dir="${OUTPUT_DIR}" \
  experiment_name="${EXPERIMENT_NAME}" \
  seed="${SEED}" \
  max_epochs="${MAX_EPOCHS}" \
  batch_size="${BATCH_SIZE}" \
  num_workers="${NUM_WORKERS}" \
  crop_size="${CROP_SIZE}" \
  overlap_size="${OVERLAP_SIZE}" \
  spacing="${SPACING}" \
  topk="${TOPK}" \
  num_samples="${NUM_SAMPLES}" \
  devices="${DEVICES}" \
  use_wandb=True \
  wandb_name="${EXPERIMENT_NAME}" \
  accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}" \
  val_full_volume=true \
  check_val_every_n_epoch="${CHECK_VAL_EVERY_N_EPOCH}" \
  val_froc_start_epoch="${VAL_FROC_START_EPOCH}" \
  val_froc_before_start_every_n_epoch="${VAL_FROC_BEFORE_START_EVERY_N_EPOCH}" \
  checkpoint_monitor=val/mean_froc \
  checkpoint_mode=max \
  checkpoint_filename='cpmnetv2_luna16_best' \
  checkpoint_every_n_epochs="${CHECK_VAL_EVERY_N_EPOCH}" \
  checkpoint_start_epoch="${CHECKPOINT_START_EPOCH}" \
  "$@"
