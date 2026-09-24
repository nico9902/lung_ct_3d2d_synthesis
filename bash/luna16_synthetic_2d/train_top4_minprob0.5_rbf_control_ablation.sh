#!/usr/bin/env bash
# Train both RBF control-point ablations in parallel, one per GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Training RBF control-point ablations."
DEVICES="${FIXED_DEVICES:-[0]}" bash "${SCRIPT_DIR}/train_top4_minprob0.5_rbf_fixed_control.sh" "$@" &
PID_FIXED=$!
DEVICES="${RANDOM_DEVICES:-[1]}" bash "${SCRIPT_DIR}/train_top4_minprob0.5_rbf_random_control.sh" "$@" &
PID_RANDOM=$!

wait "${PID_FIXED}"
wait "${PID_RANDOM}"
