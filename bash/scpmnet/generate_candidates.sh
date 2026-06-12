#!/bin/bash

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR" || exit 1

VENV_PATH="${VENV_PATH:-myenv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

CKPT="outputs/scpmnet/scpmnet_paper/checkpoints/epoch=136-val_loss=0.4910.ckpt"
CSV="/ssd2/domenico/datasets/lidc_process/lidc_labels_clean_flagged_smaller.csv"
ROOT="/ssd2/domenico/datasets/lidc_process"
OUT="outputs/scpmnet/fp_reduction"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
DEVICE="${DEVICE:-cuda:0}"

python src/det/SCPMNet/generate_candidates.py \
  --checkpoint "$CKPT" --csv-path "$CSV" --data-root "$ROOT" --batch-size 16 \
  --device "$DEVICE" \
  --split train --output-csv "$OUT/candidates_train.csv" \
  --decode-threshold 0.05 --decode-topk 300 --final-topk 300 --nms-threshold 0.05 \
  --label-candidates

python src/det/SCPMNet/generate_candidates.py \
  --checkpoint "$CKPT" --csv-path "$CSV" --data-root "$ROOT" --batch-size 16 \
  --device "$DEVICE" \
  --split val --output-csv "$OUT/candidates_val.csv" \
  --decode-threshold 0.05 --decode-topk 300 --final-topk 300 --nms-threshold 0.05 \
  --label-candidates

python src/det/SCPMNet/generate_candidates.py \
  --checkpoint "$CKPT" --csv-path "$CSV" --data-root "$ROOT" --batch-size 16 \
  --device "$DEVICE" \
  --split test --output-csv "$OUT/candidates_test.csv" \
  --decode-threshold 0.05 --decode-topk 300 --final-topk 300 --nms-threshold 0.05 \
