#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DEVICES="${DEVICES:-[2]}"
export EXPERIMENT_NAME="top3_minprob0.5_shepard"
export OUTPUT_DIR="outputs/luna16_synthetic_2d_top3_minprob0.5_shepard"
export SYNTHETIC_IMAGES_DIR="/ssd2/domenico/datasets/synthetic_2d/luna16_saliency_synthetic_detector_top3_minprob0.5_shepard"
export WANDB_PROJECT="luna16-synthetic-2d-detector-top3-minprob0.5-shepard"
export WANDB_GROUP_SUFFIX="top3_minprob0.5_shepard"

exec "${SCRIPT_DIR}/run_backbones_det_experiment.sh" "$@"
