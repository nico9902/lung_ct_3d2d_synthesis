#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Training RBF control-point ablations."
DEVICES="${FIXED_DEVICES:-[0]}" "${SCRIPT_DIR}/run_backbones_det_top4_minprob0.5_rbf_fixed_control.sh" "$@" &
PID_FIXED=$!
DEVICES="${RANDOM_DEVICES:-[1]}" "${SCRIPT_DIR}/run_backbones_det_top4_minprob0.5_rbf_random_control.sh" "$@" &
PID_RANDOM=$!

wait "${PID_FIXED}"
wait "${PID_RANDOM}"
