#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/scpmnet_luna16_10fold_normauto_amp_bs16_spv8_lr001_randomcrop_gradclip100_warmup_cosine_restarts_ep220_top3valfroc}"
export EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-scpmnet_paper_luna16_normauto_amp_bs16_spv8_lr001_gradclip100_warmup_cosine_restarts_ep220_top3valfroc_fold}"
export WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-scpmnet_luna16_amp_bs16_spv8_lr001_gradclip100_warmup_cosine_restarts_ep220_top3valfroc_fold}"

export PRECISION="${PRECISION:-16-mixed}"
export LR="${LR:-0.01}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export SAMPLES_PER_VOLUME="${SAMPLES_PER_VOLUME:-8}"
export VAL_RANDOM_CROP_SAMPLES_PER_VOLUME="${VAL_RANDOM_CROP_SAMPLES_PER_VOLUME:-4}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
export MAX_EPOCHS="${MAX_EPOCHS:-220}"

export GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-100.0}"
export GRADIENT_CLIP_ALGORITHM="${GRADIENT_CLIP_ALGORITHM:-norm}"

export CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
export CHECKPOINT_START_EPOCH="${CHECKPOINT_START_EPOCH:-100}"

export MONITOR_FINITE_VALUES="${MONITOR_FINITE_VALUES:-true}"
export FINITE_MONITOR_CHECK_LOGGED_METRICS="${FINITE_MONITOR_CHECK_LOGGED_METRICS:-true}"
export FINITE_MONITOR_CHECK_GRADIENTS="${FINITE_MONITOR_CHECK_GRADIENTS:-false}"
export FINITE_MONITOR_EVERY_N_TRAIN_STEPS="${FINITE_MONITOR_EVERY_N_TRAIN_STEPS:-1}"

export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_bs16_spv8_lr001_gradclip100_warmup_cosine_restarts_ep220_top3valfroc}"
export LOG_DIR="${LOG_DIR:-logs/scpmnet_luna16_amp_bs16_spv8_lr001_gradclip100_warmup_cosine_restarts_ep220_top3valfroc_${RUN_TAG}}"

exec bash "$SCRIPT_DIR/scpmnet_luna16_10fold_randomcrop_amp_spv4_lr003.sh" \
  lr_scheduler_name=warmup_cosine_restarts \
  warmup_epochs=20 \
  warmup_lr=0.0001 \
  cosine_restart_t_0=50 \
  cosine_restart_t_mult=1 \
  cosine_eta_min=0.000001 \
  val_froc_start_epoch=100 \
  val_froc_before_start_every_n_epoch=null \
  checkpoint_save_top_k=3 \
  posthoc_val_froc_top_k=3 \
  posthoc_val_froc_batch_size=16 \
  "$@"
