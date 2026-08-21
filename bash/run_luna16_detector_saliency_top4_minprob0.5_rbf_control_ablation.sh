#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Generating RBF control-point ablations: fixed then random."
"${SCRIPT_DIR}/run_luna16_detector_saliency_top4_minprob0.5_rbf_fixed.sh" "$@"
"${SCRIPT_DIR}/run_luna16_detector_saliency_top4_minprob0.5_rbf_random.sh" "$@"
