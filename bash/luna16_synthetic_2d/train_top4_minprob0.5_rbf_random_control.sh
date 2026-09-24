#!/usr/bin/env bash
# Train EfficientNetV2-S on the random-control RBF ablation images.
set -euo pipefail

export EXPERIMENT_NAME="cpmnetv2_bf16_top4_minprob0.50_rbf_random_control"
export WANDB_PROJECT="${WANDB_PROJECT:-luna16-synthetic-2d-guidance-ablation}"
export BACKBONES="${BACKBONES:-efficientnet_v2_s}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_backbones.sh" "$@"
