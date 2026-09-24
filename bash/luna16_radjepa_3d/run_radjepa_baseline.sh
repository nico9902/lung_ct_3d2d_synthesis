#!/usr/bin/env bash
set -euo pipefail

cd /home/domenico/lung_ct_3d2d_synthesis

# Rad-JEPA-3D baseline: use the A100 (physical GPU index 3 on this host, per
# `nvidia-smi`'s PCI-bus-ID ordering) as the sole visible device, so it
# appears as cuda:0 ("DEVICE 0"). CUDA_DEVICE_ORDER=PCI_BUS_ID is required:
# PyTorch's default CUDA enumeration does NOT match nvidia-smi's indices on
# this host (index 3 without it silently resolves to a V100).
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# Feature extraction needs mamba-ssm/causal-conv1d on torch==2.5.1+cu121,
# incompatible with this repo's myenv (torch==2.10.0). It runs with the
# sibling lung-jepa-world-model project's own working environment instead,
# by absolute path -- this repo's myenv is never modified for this baseline.
RADJEPA_PYTHON="${RADJEPA_PYTHON:-/home/domenico/lung-jepa-world-model/.venv/bin/python3}"
RADJEPA_CODE_DIR="${RADJEPA_CODE_DIR:-/home/domenico/lung-jepa-world-model/external/RadJepa}"
RADJEPA_HF_MODEL_DIR="${RADJEPA_HF_MODEL_DIR:-/home/domenico/lung-jepa-world-model/external/Rad-Jepa-3D-hf}"

SPLITS_DIR="${SPLITS_DIR:-data/processed/cv_splits}"
RAW_ROOT="${RAW_ROOT:-data/raw/LUNA16/subsets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/luna16_radjepa_3d}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-${OUTPUT_ROOT}/embeddings}"
PROBES_DIR="${PROBES_DIR:-${OUTPUT_ROOT}/probes}"
RESULTS_DIR="${RESULTS_DIR:-results/09_radjepa_foundation_baseline}"

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MONITOR="${MONITOR:-val_mcc}"
PATIENCE="${PATIENCE:-30}"
REPRESENTATIONS=("resize32" "sliding_mean" "sliding_max" "sliding_mean_max")

mkdir -p logs/luna16_radjepa_3d "${EMBEDDINGS_DIR}" "${PROBES_DIR}" "${RESULTS_DIR}"
LOG_FILE="logs/luna16_radjepa_3d/$(date +%Y%m%d_%H%M%S)_radjepa_baseline.log"

{
  echo "Logging to ${LOG_FILE}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER})"
  echo "RADJEPA_PYTHON=${RADJEPA_PYTHON}"
  echo "EMBEDDINGS_DIR=${EMBEDDINGS_DIR}"
  echo "PROBES_DIR=${PROBES_DIR}"

  echo "===== Step 1/3: extracting frozen Rad-JEPA-3D embeddings (cached, run once) ====="
  "${RADJEPA_PYTHON}" src/luna16_radjepa_3d/extract_features.py \
    --fold-csv "${SPLITS_DIR}/luna16_classification_fold0.csv" \
    --raw-root "${RAW_ROOT}" \
    --output-dir "${EMBEDDINGS_DIR}" \
    --radjepa-code-dir "${RADJEPA_CODE_DIR}" \
    --hf-model-dir "${RADJEPA_HF_MODEL_DIR}" \
    --device cuda

  echo "===== Step 2/3: training frozen linear probes for 10 folds x 4 representations ====="
  source myenv/bin/activate
  export PYTHONPATH="/home/domenico/lung_ct_3d2d_synthesis:${PYTHONPATH:-}"
  for REPRESENTATION in "${REPRESENTATIONS[@]}"; do
    for FOLD in 0 1 2 3 4 5 6 7 8 9; do
      echo "--- representation=${REPRESENTATION} fold=${FOLD} ---"
      python src/luna16_radjepa_3d/train_linear_probe.py \
        --embeddings-dir "${EMBEDDINGS_DIR}/per_patient" \
        --splits-dir "${SPLITS_DIR}" \
        --output-dir "${PROBES_DIR}" \
        --fold "${FOLD}" \
        --representation "${REPRESENTATION}" \
        --batch-size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --monitor "${MONITOR}" \
        --patience "${PATIENCE}" \
        --device cuda
    done
  done

  echo "===== Step 3/3: aggregating pooled + per-fold mean/std metrics ====="
  python src/luna16_radjepa_3d/aggregate.py \
    --output-dir "${PROBES_DIR}" \
    --representations "${REPRESENTATIONS[@]}" \
    --report-name "radjepa_pooled_results.md"

  cp "${PROBES_DIR}/radjepa_pooled_results.md" "${RESULTS_DIR}/radjepa_pooled_results.md"
  cp "${PROBES_DIR}/pooled_metrics.csv" "${RESULTS_DIR}/pooled_metrics.csv"
  cp "${PROBES_DIR}/per_fold_mean_std_metrics.csv" "${RESULTS_DIR}/per_fold_mean_std_metrics.csv"

  echo "Done. Report copied to ${RESULTS_DIR}/radjepa_pooled_results.md"
} 2>&1 | tee "${LOG_FILE}"
