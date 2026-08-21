#!/usr/bin/env bash
set -euo pipefail

cd /home/domenico/lung_ct_3d2d_synthesis
source myenv/bin/activate

export PYTHONPATH="/home/domenico/lung_ct_3d2d_synthesis:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/ssd2/domenico/datasets/LUNA16_preprocessed}"
SPLITS_DIR="${SPLITS_DIR:-/ssd2/domenico/datasets/LUNA16_preprocessed/cv_splits}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/luna16_slice_attention_2p5d_all_slices_256x384_effnetv2s}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-256}"
IMAGE_WIDTH="${IMAGE_WIDTH:-384}"
ENCODER_CHUNK_SIZE="${ENCODER_CHUNK_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
EPOCHS="${EPOCHS:-100}"
PRECISION="${PRECISION:-16-mixed}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-[0]}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-luna16-slice-attention-2p5d}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"

mkdir -p logs/luna16_slice_attention_2p5d

EXTRA_ARGS=()
if [[ "${WANDB}" == "1" ]]; then
  EXTRA_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}")
fi
if [[ "${WANDB_OFFLINE}" == "1" ]]; then
  EXTRA_ARGS+=(--wandb-offline)
fi

echo "Training LUNA16 2.5D all-slice attention baseline"
echo "  Output: ${OUTPUT_DIR}"
echo "  Image size: ${IMAGE_HEIGHT}x${IMAGE_WIDTH}"
echo "  Encoder chunk size: ${ENCODER_CHUNK_SIZE}"
echo "  Batch size / grad accumulation: ${BATCH_SIZE}/${ACCUMULATE_GRAD_BATCHES}"
echo "  Folds: ${FOLDS}"

for FOLD in ${FOLDS}; do
  python3 -m src.luna16_slice_attention_2p5d.train \
    --output-dir "${OUTPUT_DIR}" \
    --data-root "${DATA_ROOT}" \
    --splits-dir "${SPLITS_DIR}" \
    --fold "${FOLD}" \
    --backbone efficientnet_v2_s \
    --image-size "${IMAGE_HEIGHT}" "${IMAGE_WIDTH}" \
    --encoder-chunk-size "${ENCODER_CHUNK_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --accumulate-grad-batches "${ACCUMULATE_GRAD_BATCHES}" \
    --epochs "${EPOCHS}" \
    --precision "${PRECISION}" \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --num-workers "${NUM_WORKERS}" \
    --monitor val_mcc \
    --wandb-group "efficientnet_v2_s_all_slices_2p5d_attention" \
    "${EXTRA_ARGS[@]}"
done

python3 -m src.luna16_slice_attention_2p5d.aggregate \
  --output-dir "${OUTPUT_DIR}" \
  --backbone efficientnet_v2_s
