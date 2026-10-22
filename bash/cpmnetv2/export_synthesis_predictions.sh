#!/usr/bin/env bash
# Convert detector test boxes into the candidate tables used by downstream experiments.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
python -m src.det.CPMNetv2.export_synthesis_predictions \
  --input-root "${DETECTOR_OUTPUT_DIR:-outputs/cpmnetv2_luna16_10fold_bf16_guarded}" \
  --output-root "${DETECTOR_PREDICTIONS}" \
  --splits-dir "${SPLITS_DIR}" \
  --folds ${FOLDS} \
  --pooled-csv "${POOLED_DETECTOR_CSV:-outputs/cpmnetv2_paper_detector_all_oof_predictions.csv}" \
  "$@"
