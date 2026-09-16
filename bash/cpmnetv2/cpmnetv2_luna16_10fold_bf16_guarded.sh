#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-/ssd2/domenico/datasets/LUNA16_preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cpmnetv2_luna16_10fold_bf16_guarded}"
BASE_TAG="${BASE_TAG:-20260809_cpmnetv2_luna16_10fold_bs8_numsam3_topk7_a100_lrbase0001_lrmax001_bf16_guarded}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"

mkdir -p logs/cpmnetv2

for FOLD in $FOLDS; do
  SPLIT_CSV="$DATA_ROOT/cv_splits/luna16_fold${FOLD}.csv"
  EXPERIMENT_NAME="${BASE_TAG}_fold${FOLD}"
  FOLD_LOG="logs/cpmnetv2/${EXPERIMENT_NAME}.log"

  if [ ! -f "$SPLIT_CSV" ]; then
    echo "Missing split CSV: $SPLIT_CSV" >&2
    exit 1
  fi

  if [ -f "$OUTPUT_DIR/$EXPERIMENT_NAME/checkpoints/cpmnetv2_luna16_best.ckpt" ]; then
    echo "Skipping fold ${FOLD}: checkpoint already exists."
    continue
  fi

  echo "Starting fold ${FOLD}: ${EXPERIMENT_NAME}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  DATA_ROOT="$DATA_ROOT" \
  SPLIT_CSV="$SPLIT_CSV" \
  LABELS_CSV="$SPLIT_CSV" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  BATCH_SIZE="${BATCH_SIZE:-8}" \
  NUM_SAMPLES="${NUM_SAMPLES:-3}" \
  TOPK="${TOPK:-7}" \
  MAX_EPOCHS="${MAX_EPOCHS:-170}" \
  VAL_FROC_START_EPOCH="${VAL_FROC_START_EPOCH:-100}" \
  VAL_FROC_BEFORE_START_EVERY_N_EPOCH="${VAL_FROC_BEFORE_START_EVERY_N_EPOCH:-100}" \
  CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-5}" \
  CHECKPOINT_START_EPOCH="${CHECKPOINT_START_EPOCH:-100}" \
  NUM_WORKERS="${NUM_WORKERS:-8}" \
  DEVICES="${DEVICES:-[0]}" \
  ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}" \
  bash bash/cpmnetv2/cpmnetv2_luna16_holdout_val_froc.sh \
    precision=bf16-mixed \
    accelerator=gpu \
    use_wandb=True \
    wandb_project=lung_ct_3d2d_synthesis_detection \
    wandb_name="$EXPERIMENT_NAME" \
    lr=0.001 \
    warmup_multiplier=10.0 \
    warmup_epochs=2 \
    cosine_t_max=170 \
    eta_min=1e-6 \
    > "$FOLD_LOG" 2>&1

  echo "Finished fold ${FOLD}: ${EXPERIMENT_NAME}"
done
