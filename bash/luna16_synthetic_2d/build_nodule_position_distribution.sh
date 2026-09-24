#!/usr/bin/env bash
# Build the empirical nodule position/count distribution used to place pseudo
# nodules when the detector finds no candidate (--use-empirical-pseudo-nodules
# in generate_synthetic_images.sh). It is estimated from the control points of
# the ground-truth-guided synthetic images in GT_SYNTHETIC_ROOT.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

GT_SYNTHETIC_ROOT="${GT_SYNTHETIC_ROOT:-outputs/luna16_saliency_synthetic_gt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/luna16_saliency_control_point_distribution}"
LABELS_CSV="${LABELS_CSV:-${DATA_ROOT}/luna16_labels.csv}"
# Optional GT saliency log; without it positive scans are taken from LABELS_CSV.
LOG_PATH="${LOG_PATH:-luna16_synthetic_2d.txt}"

python -m src.luna16_synthetic_2d.build_nodule_position_distribution \
  --source control-points \
  --output-root "${GT_SYNTHETIC_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --labels-csv "${LABELS_CSV}" \
  --preprocessed-root "${DATA_ROOT}" \
  --log-path "${LOG_PATH}" \
  "$@"
