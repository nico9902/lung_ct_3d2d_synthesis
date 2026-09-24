#!/usr/bin/env bash
# Generate both RBF control-point ablations (fixed, then random).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/generate_top4_minprob0.5_rbf_fixed_control.sh" "$@"
bash "${SCRIPT_DIR}/generate_top4_minprob0.5_rbf_random_control.sh" "$@"
