#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TOP_K="3"
export MIN_PROBABILITY="0.5"
export SURFACE_METHOD="rbf"
export OUTPUT_ROOT="data/luna16_saliency_synthetic_detector_top3_minprob0.5_rbf"

exec "${SCRIPT_DIR}/run_luna16_detector_saliency_top7_minprob0.3_rbf.sh" "$@"
