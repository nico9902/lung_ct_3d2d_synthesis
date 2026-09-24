#!/usr/bin/env bash
# Geometric ablation: RBF surface with fixed control points.
set -euo pipefail

export TOP_K="4"
export MIN_PROBABILITY="0.5"
export SURFACE_METHOD="rbf"
export CONTROL_POINT_MODE="fixed"
export EXPERIMENT_NAME="cpmnetv2_bf16_top4_minprob0.50_rbf_fixed_control"

exec bash "$(dirname "${BASH_SOURCE[0]}")/generate_synthetic_images.sh" "$@"
