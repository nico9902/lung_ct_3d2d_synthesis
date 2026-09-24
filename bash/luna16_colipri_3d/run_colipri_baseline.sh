#!/usr/bin/env bash
set -euo pipefail

cd /home/domenico/lung_ct_3d2d_synthesis
source myenv/bin/activate

# COLIPRI-CRM baseline: use the A100 (physical GPU index 3 on this host, per
# `nvidia-smi`'s PCI-bus-ID ordering) as the sole visible device, so it appears
# as cuda:0 ("DEVICE 0") to the scripts. CUDA_DEVICE_ORDER=PCI_BUS_ID is
# required: PyTorch's default CUDA enumeration does NOT match nvidia-smi's
# indices on this host (index 3 without it silently resolves to a V100).
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONPATH="/home/domenico/lung_ct_3d2d_synthesis:${PYTHONPATH:-}"

SPLITS_DIR="${SPLITS_DIR:-data/processed/cv_splits}"
RAW_ROOT="${RAW_ROOT:-data/raw/LUNA16/subsets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/luna16_colipri_3d}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-${OUTPUT_ROOT}/embeddings}"
PROBES_DIR="${PROBES_DIR:-${OUTPUT_ROOT}/probes}"
RESULTS_DIR="${RESULTS_DIR:-results/08_colipri_foundation_baseline}"

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MONITOR="${MONITOR:-val_mcc}"
PATIENCE="${PATIENCE:-30}"
EXTRACT_NUM_WORKERS="${EXTRACT_NUM_WORKERS:-12}"
REPRESENTATIONS=("pooled" "dense_maxpool")

mkdir -p logs/luna16_colipri_3d "${EMBEDDINGS_DIR}" "${PROBES_DIR}" "${RESULTS_DIR}"
LOG_FILE="logs/luna16_colipri_3d/$(date +%Y%m%d_%H%M%S)_colipri_baseline.log"

{
  echo "Logging to ${LOG_FILE}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "EMBEDDINGS_DIR=${EMBEDDINGS_DIR}"
  echo "PROBES_DIR=${PROBES_DIR}"

  echo "===== Step 1/3: extracting frozen COLIPRI-CRM embeddings (cached, run once) ====="
  python src/luna16_colipri_3d/extract_features.py \
    --fold-csv "${SPLITS_DIR}/luna16_classification_fold0.csv" \
    --raw-root "${RAW_ROOT}" \
    --output-dir "${EMBEDDINGS_DIR}" \
    --num-workers "${EXTRACT_NUM_WORKERS}" \
    --device cuda

  echo "===== Step 2/3: training frozen linear probes for 10 folds x 2 representations ====="
  for REPRESENTATION in "${REPRESENTATIONS[@]}"; do
    for FOLD in 0 1 2 3 4 5 6 7 8 9; do
      echo "--- representation=${REPRESENTATION} fold=${FOLD} ---"
      python src/luna16_colipri_3d/train_linear_probe.py \
        --embeddings "${EMBEDDINGS_DIR}/colipri_embeddings.npz" \
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
  python src/luna16_colipri_3d/aggregate.py \
    --output-dir "${PROBES_DIR}" \
    --representations "${REPRESENTATIONS[@]}" \
    --report-name "colipri_pooled_results.md"

  cp "${PROBES_DIR}/colipri_pooled_results.md" "${RESULTS_DIR}/colipri_pooled_results.md"
  cp "${PROBES_DIR}/pooled_metrics.csv" "${RESULTS_DIR}/pooled_metrics.csv"
  cp "${PROBES_DIR}/per_fold_mean_std_metrics.csv" "${RESULTS_DIR}/per_fold_mean_std_metrics.csv"

  echo "Done. Report copied to ${RESULTS_DIR}/colipri_pooled_results.md"
} 2>&1 | tee "${LOG_FILE}"
