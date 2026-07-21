#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNS=(
  "run_luna16_detector_saliency_top4_minprob0.5_rbf.sh"
  "run_luna16_detector_saliency_top4_minprob0.5_shepard.sh"
  "run_luna16_detector_saliency_top3_minprob0.5_rbf.sh"
  "run_luna16_detector_saliency_top3_minprob0.5_shepard.sh"
)

for RUN_SCRIPT in "${RUNS[@]}"; do
  echo "========================================"
  echo "Running ${RUN_SCRIPT}"
  echo "========================================"
  "${SCRIPT_DIR}/${RUN_SCRIPT}" "$@"
done
