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
DETECTOR_OUTPUT_ROOT="${DETECTOR_OUTPUT_ROOT:-outputs/scpmnet_luna16_10fold}"
DETECTOR_FOLD_GLOB="${DETECTOR_FOLD_GLOB:-scpmnet_paper_luna16_fold*}"
DETECTOR_PREDICTION_NAME="${DETECTOR_PREDICTION_NAME:-test_predictions.csv}"
FPR_OUTPUT_ROOT="${FPR_OUTPUT_ROOT:-outputs/scpmnet_luna16_10fold_fpr_top100_focal_balanced_average_normauto}"

START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-9}"
TOP_CANDIDATES_PER_VOLUME="${TOP_CANDIDATES_PER_VOLUME:-100}"
GENERATE_DETECTIONS="${GENERATE_DETECTIONS:-false}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-16-mixed}"

GENERATE_BATCH_SIZE="${GENERATE_BATCH_SIZE:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
RESCORE_BATCH_SIZE="${RESCORE_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VOLUME_CACHE_SIZE="${VOLUME_CACHE_SIZE:-4}"
NORMALIZED_VOLUME_CACHE_DIR="${NORMALIZED_VOLUME_CACHE_DIR:-$FPR_OUTPUT_ROOT/normalized_volume_cache_${INTENSITY_MODE:-auto}}"

DECODE_THRESHOLD="${DECODE_THRESHOLD:-0.05}"
DECODE_TOPK="${DECODE_TOPK:-300}"
NMS_THRESHOLD="${NMS_THRESHOLD:-0.05}"
IGNORE_MARGIN="${IGNORE_MARGIN:-2.0}"

MAX_EPOCHS="${MAX_EPOCHS:-100}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-10000}"
LOSS="${LOSS:-focal}"
FOCAL_ALPHA="${FOCAL_ALPHA:-0.5}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
INTENSITY_MODE="${INTENSITY_MODE:-auto}"
NO_BALANCED_SAMPLER="${NO_BALANCED_SAMPLER:-false}"
POS_WEIGHT="${POS_WEIGHT:-none}"
SCORE_MODE="${SCORE_MODE:-average}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lung_ct_3d2d_synthesis_detection_fpr}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

TRAIN_SCRIPT="$PROJECT_DIR/src/det/SCPMNet/train_fp_reduction.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "Missing FPR training script: $TRAIN_SCRIPT" >&2
  exit 1
fi
if [ "$LOSS" != "bce" ] && ! grep -q -- "--loss" "$TRAIN_SCRIPT"; then
  echo "The selected train_fp_reduction.py does not support --loss." >&2
  echo "Update $TRAIN_SCRIPT before running LOSS=$LOSS." >&2
  exit 1
fi

find_detector_checkpoint() {
  local fold_dir="$1"
  local best_ckpt
  best_ckpt="$(find "$fold_dir/checkpoints" -maxdepth 1 -name '*.ckpt' ! -name 'last.ckpt' | sort | tail -n 1)"
  if [ -n "$best_ckpt" ]; then
    echo "$best_ckpt"
  else
    echo "$fold_dir/checkpoints/last.ckpt"
  fi
}

find_fpr_checkpoint() {
  local fpr_dir="$1"
  find "$fpr_dir/checkpoints" -maxdepth 1 -name '*.ckpt' | sort | tail -n 1
}

for FOLD in $(seq "$START_FOLD" "$END_FOLD"); do
  CSV_PATH="$SPLIT_DIR/luna16_fold${FOLD}.csv"
  DETECTOR_FOLD_DIR="$DETECTOR_OUTPUT_ROOT/scpmnet_paper_luna16_fold${FOLD}"
  FOLD_OUT="$FPR_OUTPUT_ROOT/scpmnet_paper_luna16_fold${FOLD}"
  FPR_DIR="$FOLD_OUT/fp_reduction"
  PRED_DIR="$FOLD_OUT/predictions"
  CKPT=""
  if [ "$GENERATE_DETECTIONS" = "true" ]; then
    CKPT="$(find_detector_checkpoint "$DETECTOR_FOLD_DIR")"
  fi

  mkdir -p "$FPR_DIR" "$PRED_DIR"

  if [ "$GENERATE_DETECTIONS" = "true" ]; then
    echo "==== Fold $FOLD | detector checkpoint: $CKPT"
  else
    echo "==== Fold $FOLD | using existing detector predictions from: $DETECTOR_OUTPUT_ROOT"
  fi
  echo "==== Fold $FOLD | Top-$TOP_CANDIDATES_PER_VOLUME candidates per volume"

  for SPLIT in train val test; do
    LABEL_ARGS=()
    if [ "$SPLIT" != "test" ]; then
      LABEL_ARGS=(--label-candidates)
    fi

    if [ "$GENERATE_DETECTIONS" = "true" ]; then
      python -m src.det.SCPMNet.generate_candidates \
        --checkpoint "$CKPT" \
        --csv-path "$CSV_PATH" \
        --data-root "$DATA_ROOT" \
        --split "$SPLIT" \
        --output-csv "$FPR_DIR/candidates_${SPLIT}_top${TOP_CANDIDATES_PER_VOLUME}.csv" \
        --batch-size "$GENERATE_BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --device "$DEVICE" \
        --decode-threshold "$DECODE_THRESHOLD" \
        --decode-topk "$DECODE_TOPK" \
        --top-candidates-per-volume "$TOP_CANDIDATES_PER_VOLUME" \
        --nms-threshold "$NMS_THRESHOLD" \
        --ignore-margin "$IGNORE_MARGIN" \
        "${LABEL_ARGS[@]}"
    else
      python -m src.det.SCPMNet.build_fpr_candidates_from_predictions \
        --prediction-root "$DETECTOR_OUTPUT_ROOT" \
        --fold-glob "$DETECTOR_FOLD_GLOB" \
        --prediction-name "$DETECTOR_PREDICTION_NAME" \
        --csv-path "$CSV_PATH" \
        --data-root "$DATA_ROOT" \
        --split "$SPLIT" \
        --output-csv "$FPR_DIR/candidates_${SPLIT}_top${TOP_CANDIDATES_PER_VOLUME}.csv" \
        --top-candidates-per-volume "$TOP_CANDIDATES_PER_VOLUME" \
        --nms-threshold "$NMS_THRESHOLD" \
        --ignore-margin "$IGNORE_MARGIN" \
        "${LABEL_ARGS[@]}"
    fi
  done

  TRAIN_ARGS=()
  SAMPLER_TAG="balanced"
  if [ "$NO_BALANCED_SAMPLER" = "true" ]; then
    TRAIN_ARGS+=(--no-balanced-sampler)
    SAMPLER_TAG="natural"
  fi
  if [ -n "$POS_WEIGHT" ]; then
    TRAIN_ARGS+=(--pos-weight "$POS_WEIGHT")
  fi
  if [ -n "$SAMPLES_PER_EPOCH" ]; then
    TRAIN_ARGS+=(--samples-per-epoch "$SAMPLES_PER_EPOCH")
  fi
  if [ "$USE_WANDB" = "true" ]; then
    TRAIN_ARGS+=(
      --use-wandb
      --wandb-project "$WANDB_PROJECT"
      --wandb-name "luna16_fpr_top${TOP_CANDIDATES_PER_VOLUME}_${LOSS}_${SAMPLER_TAG}_${SCORE_MODE}_${INTENSITY_MODE}_fold${FOLD}"
      --wandb-mode "$WANDB_MODE"
    )
    if [ -n "$WANDB_ENTITY" ]; then
      TRAIN_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    fi
  fi

  python "$TRAIN_SCRIPT" \
    --train-candidates "$FPR_DIR/candidates_train_top${TOP_CANDIDATES_PER_VOLUME}.csv" \
    --val-candidates "$FPR_DIR/candidates_val_top${TOP_CANDIDATES_PER_VOLUME}.csv" \
    --csv-path "$CSV_PATH" \
    --data-root "$DATA_ROOT" \
    --output-dir "$FPR_DIR" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --volume-cache-size "$VOLUME_CACHE_SIZE" \
    --normalized-volume-cache-dir "$NORMALIZED_VOLUME_CACHE_DIR" \
    --max-epochs "$MAX_EPOCHS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --loss "$LOSS" \
    --focal-alpha "$FOCAL_ALPHA" \
    --focal-gamma "$FOCAL_GAMMA" \
    --intensity-mode "$INTENSITY_MODE" \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES" \
    --precision "$PRECISION" \
    "${TRAIN_ARGS[@]}"

  FP_CKPT="$(find_fpr_checkpoint "$FPR_DIR")"
  if [ -z "$FP_CKPT" ]; then
    echo "No FPR checkpoint found for fold $FOLD in $FPR_DIR/checkpoints" >&2
    exit 1
  fi
  echo "==== Fold $FOLD | FPR checkpoint: $FP_CKPT"

  python -m src.det.SCPMNet.rescore_candidates \
    --classifier-checkpoint "$FP_CKPT" \
    --candidates "$FPR_DIR/candidates_test_top${TOP_CANDIDATES_PER_VOLUME}.csv" \
    --csv-path "$CSV_PATH" \
    --data-root "$DATA_ROOT" \
    --split test \
    --output-dir "$PRED_DIR" \
    --patch-size 32 32 32 \
    --batch-size "$RESCORE_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --volume-cache-size "$VOLUME_CACHE_SIZE" \
    --normalized-volume-cache-dir "$NORMALIZED_VOLUME_CACHE_DIR" \
    --device "$DEVICE" \
    --intensity-mode "$INTENSITY_MODE" \
    --score-mode "$SCORE_MODE"
done

python -m src.det.SCPMNet.aggregate_luna16_cv \
  --output-root "$FPR_OUTPUT_ROOT" \
  --split-dir "$SPLIT_DIR" \
  --prediction-name test_predictions_rescored.csv \
  --froc-name test_froc_rescored.csv \
  --out-dir "$FPR_OUTPUT_ROOT/cv_aggregate"

echo "FPR Top-$TOP_CANDIDATES_PER_VOLUME completed."
echo "Use rescored predictions with:"
echo "  --pred-root $FPR_OUTPUT_ROOT --prediction-name test_predictions_rescored.csv"
