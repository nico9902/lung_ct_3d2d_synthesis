#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DEVICES="${DEVICES:-[0]}"
export EXPERIMENT_NAME="top4_minprob0.5_rbf_fixed_control"
export OUTPUT_DIR="outputs/luna16_synthetic_2d_top4_minprob0.5_rbf_fixed_control"
export SYNTHETIC_IMAGES_DIR="data/luna16_saliency_synthetic_detector_top4_minprob0.5_rbf_fixed_control"
export WANDB_PROJECT="luna16-synthetic-2d-guidance-ablation"
export WANDB_GROUP_SUFFIX="top4_minprob0.5_rbf_fixed_control"
export BACKBONES="${BACKBONES:-efficientnet_v2_s}"

exec "${SCRIPT_DIR}/run_backbones_det_experiment.sh" "$@"
