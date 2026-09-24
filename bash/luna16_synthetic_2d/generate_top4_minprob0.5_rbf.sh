#!/usr/bin/env bash
# Proposed method: detector-guided adaptive RBF surface (top-4, p >= 0.5).
set -euo pipefail

export TOP_K="4"
export MIN_PROBABILITY="0.5"
export SURFACE_METHOD="rbf"
export CONTROL_POINT_MODE="detector"
export EXPERIMENT_NAME="cpmnetv2_bf16_top4_minprob0.50_rbf"

exec bash "$(dirname "${BASH_SOURCE[0]}")/generate_synthetic_images.sh" "$@"
