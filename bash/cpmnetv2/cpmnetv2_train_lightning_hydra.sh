#!/bin/bash

# ==============================================================================
# GravitySpace Detection Training - Subvolume Memory Optimized
# ==============================================================================

PROJECT_DIR="/home/domenico/lung_ct_3d2d_synthesis"
cd "$PROJECT_DIR" || exit 1

VENV_PATH="myenv"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

echo "Activating virtual environment..."
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
elif [ -f "venv/bin/activate" ]; then
    echo "Warning: '$VENV_PATH' not found, falling back to 'venv'..."
    source venv/bin/activate
else
    echo "Error: Virtual environment not found."
    exit 1
fi

# Keep these defaults mirrored with src/det/CPMNetv2/conf/train_lightning.yaml.
CSV_PATH="data/dataset_nodule_mean.csv"
IMAGES_DIR="/ssd2/domenico/datasets/lidc_process"
ANNOTATIONS_DIR="/ssd2/domenico/datasets/lidc_process"
LABELS_CSV="/ssd2/domenico/datasets/lidc_process/lidc_labels.csv"
OUTPUT_DIR="outputs/cpmnetv2"
EXPERIMENT_NAME="cpmnetv2"
USE_WANDB="true"
WANDB_PROJECT="lung_ct_3d2d_synthesis_detection"
WANDB_NAME="cpmnetv2"
WANDB_ENTITY="domenico-paolo1999"
WANDB_MODE="online"
WANDB_LOG_MODEL="all"

VIEW="axial"
CHECKPOINT="outputs/cpmnetv2/cpmnetv2/checkpoints/epoch\=078-val_loss\=0.0000.ckpt"
TEST_ONLY="false"
NO_FROC="false"
FROC_IOU_THRESHOLD="0.1"

SEED="233"
BATCH_SIZE="2"
NUM_WORKERS="8"
MAX_EPOCHS="150"
ACCELERATOR="auto"
DEVICES="[3]"
PRECISION="bf16-mixed"  # Use "16" for older GPUs without native bfloat16 support
LOG_EVERY_N_STEPS="10"
VAL_CHECK_INTERVAL="null"
ACCUMULATE_GRAD_BATCHES="8"
NUM_SANITY_VAL_STEPS="0"

LR="0.01"
TOPK="15"
NUM_SAMPLES="2"
LAMBDA_CLS="4.0"
LAMBDA_OFFSET="1.0"
LAMBDA_SHAPE="0.1"
LAMBDA_IOU="1.0"
NORM_TYPE="batchnorm"
HEAD_NORM="batchnorm"
ACT_TYPE="ReLU"
SE="false"
POST_THRESHOLD="0.15"
CONFIDENCE_LOG_INTERVAL="50"
DEBUG_TARGET_STATS="true"
DEBUG_TARGET_STATS_INTERVAL="10"

CROP_SIZE="[96, 96, 96]"
OVERLAP_SIZE="[24, 24, 24]"
SPACING="[1.0, 1.0, 1.0]"
MIN_COMPONENT_VOXELS="1"

python -m src.det.CPMNetv2.train_lightning \
  csv_path="$CSV_PATH" \
  images_dir="$IMAGES_DIR" \
  annotations_dir="$ANNOTATIONS_DIR" \
  labels_csv="$LABELS_CSV" \
  output_dir="$OUTPUT_DIR" \
  experiment_name="$EXPERIMENT_NAME" \
  use_wandb="$USE_WANDB" \
  wandb_project="$WANDB_PROJECT" \
  wandb_name="$WANDB_NAME" \
  wandb_entity="$WANDB_ENTITY" \
  wandb_mode="$WANDB_MODE" \
  wandb_log_model="$WANDB_LOG_MODEL" \
  view="$VIEW" \
  checkpoint="$CHECKPOINT" \
  test_only="$TEST_ONLY" \
  no_froc="$NO_FROC" \
  froc_iou_threshold="$FROC_IOU_THRESHOLD" \
  seed="$SEED" \
  batch_size="$BATCH_SIZE" \
  num_workers="$NUM_WORKERS" \
  max_epochs="$MAX_EPOCHS" \
  accelerator="$ACCELERATOR" \
  devices="$DEVICES" \
  precision="$PRECISION" \
  log_every_n_steps="$LOG_EVERY_N_STEPS" \
  val_check_interval="$VAL_CHECK_INTERVAL" \
  accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
  num_sanity_val_steps="$NUM_SANITY_VAL_STEPS" \
  lr="$LR" \
  topk="$TOPK" \
  num_samples="$NUM_SAMPLES" \
  lambda_cls="$LAMBDA_CLS" \
  lambda_offset="$LAMBDA_OFFSET" \
  lambda_shape="$LAMBDA_SHAPE" \
  lambda_iou="$LAMBDA_IOU" \
  norm_type="$NORM_TYPE" \
  head_norm="$HEAD_NORM" \
  act_type="$ACT_TYPE" \
  se="$SE" \
  post_threshold="$POST_THRESHOLD" \
  confidence_log_interval="$CONFIDENCE_LOG_INTERVAL" \
  debug_target_stats="$DEBUG_TARGET_STATS" \
  debug_target_stats_interval="$DEBUG_TARGET_STATS_INTERVAL" \
  crop_size="$CROP_SIZE" \
  overlap_size="$OVERLAP_SIZE" \
  spacing="$SPACING" \
  min_component_voxels="$MIN_COMPONENT_VOXELS" \
  "$@"
