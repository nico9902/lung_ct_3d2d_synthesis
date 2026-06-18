#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/luna16_synthetic_2d}"
MANIFEST_CSV="${MANIFEST_CSV:-outputs/scpmnet_luna16_10fold_tps_images/manifest.csv}"
SPLITS_DIR="${SPLITS_DIR:-ssd2/domenico/datasets/LUNA16_preprocessed/cv_splits}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
BACKBONES="${BACKBONES:-vgg16 efficientnet_b0 efficientnet_b1 resnet18 resnet50 densenet121}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-24}"
PRECISION="${PRECISION:-bf16-mixed}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-256}"
IMAGE_WIDTH="${IMAGE_WIDTH:-384}"

for FOLD in ${FOLDS}; do
  for BACKBONE in ${BACKBONES}; do
    python3 -m src.luna16_synthetic_2d.train \
      --output-dir "${OUTPUT_DIR}" \
      --manifest-csv "${MANIFEST_CSV}" \
      --splits-dir "${SPLITS_DIR}" \
      --fold "${FOLD}" \
      --backbone "${BACKBONE}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --image-size "${IMAGE_HEIGHT}" "${IMAGE_WIDTH}" \
      --precision "${PRECISION}"
  done
done
