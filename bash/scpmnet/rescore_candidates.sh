#!/bin/bash

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR" || exit 1

VENV_PATH="${VENV_PATH:-myenv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

CKPT="outputs/scpmnet/scpmnet_paper/checkpoints/epoch=136-val_loss\=0.4910.ckpt"
CSV="/ssd2/domenico/datasets/lidc_process/lidc_labels_clean_flagged_smaller.csv"
ROOT="/ssd2/domenico/datasets/lidc_process"
OUT="${OUT:-outputs/scpmnet/scpmnet_paper/fp_reduction}"
WANDB_PROJECT="${WANDB_PROJECT:-lung_ct_3d2d_synthesis_detection}"
WANDB_NAME="${WANDB_NAME:-scpmnet_fp_reduction}"
WANDB_MODE="${WANDB_MODE:-online}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
DEVICE="${DEVICE:-cuda:0}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-bf16-mixed}"

FP_CKPT="${FP_CKPT:-$OUT/checkpoints/epoch=025-val_loss=0.0855.ckpt}"

python src/det/SCPMNet/rescore_candidates.py \
  --classifier-checkpoint "$FP_CKPT" \
  --candidates "$OUT/candidates_test.csv" \
  --csv-path "$CSV" --data-root "$ROOT" \
  --split test \
  --output-dir "$OUT/rescored_test" \
  --device "$DEVICE" \
  --batch-size "${RESCORE_BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --volume-cache-size "${VOLUME_CACHE_SIZE:-4}" \
  --score-mode multiply
