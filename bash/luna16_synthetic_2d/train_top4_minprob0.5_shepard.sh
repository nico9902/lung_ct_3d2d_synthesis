#!/usr/bin/env bash
# Train all 2D backbones on the adaptive Shepard images.
set -euo pipefail

export EXPERIMENT_NAME="cpmnetv2_bf16_top4_minprob0.50_shepard"
export WANDB_PROJECT="${WANDB_PROJECT:-luna16-synthetic-2d-detector-top4-minprob0.5-shepard}"
# Directory of the 10-fold run reported in the paper.
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/luna16_synthetic_2d_${EXPERIMENT_NAME}_v100}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_backbones.sh" "$@"
