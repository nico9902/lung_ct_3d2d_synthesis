#!/bin/bash

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR" || exit 1

VENV_PATH="${VENV_PATH:-myenv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

CSV_PATH="${CSV_PATH:-/ssd2/domenico/datasets/lidc_process/lidc_labels_clean_flagged_smaller.csv}"
DATA_ROOT="${DATA_ROOT:-/ssd2/domenico/datasets/lidc_process}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/scpmnet}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-scpmnet_paper}"

python -m src.det.SCPMNet.train_lightning \
  --config-name train_lightning_paper \
  csv_path="$CSV_PATH" \
  data_root="$DATA_ROOT" \
  output_dir="$OUTPUT_DIR" \
  experiment_name="$EXPERIMENT_NAME" \
  batch_size=8 \
  devices=[3] \
  use_wandb=True \
  samples_per_volume=1 \
  accumulate_grad_batches=3 \
  checkpoint="outputs/scpmnet/scpmnet_paper/checkpoints/epoch\=136-val_loss\=0.4910.ckpt" \
  test_only=true \
  decode_threshold=0.05 \
  decode_topk=300 \
  final_topk=300 \
  nms_threshold=0.05 \
  "$@"
