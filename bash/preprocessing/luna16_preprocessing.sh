#!/usr/bin/env bash
# Resample LUNA16 .mhd scans to isotropic spacing, crop to the lungs and write
# per-scan NIfTI volumes, lung/nodule masks and the detector labels CSV.
#
# Set LUNG_MASK_ONLY=1 to only (re)write lung masks for already preprocessed scans.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

LUNA16_ROOT="${LUNA16_ROOT:-data/raw/LUNA16}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}}"
LABELS_CSV="${LABELS_CSV:-${OUTPUT_DIR}/luna16_labels.csv}"
SPACING="${SPACING:-1 1 1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
LUNG_MASK_ONLY="${LUNG_MASK_ONLY:-0}"

EXTRA_ARGS=()
if [[ "${LUNG_MASK_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--lung-mask-only)
fi

python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

python -m src.prs.luna16_preprocessing \
  --luna-root "${LUNA16_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --labels-csv "${LABELS_CSV}" \
  --spacing ${SPACING} \
  --num-workers "${NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

echo "Preprocessing completed: ${OUTPUT_DIR}"
