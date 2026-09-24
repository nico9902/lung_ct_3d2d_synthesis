#!/usr/bin/env bash
# Shared setup sourced by every launcher in bash/.
#
# - resolves the repository root and cds into it, so launchers can be run
#   from any working directory;
# - activates a virtual environment (VENV_PATH, then myenv/.venv/venv);
# - exports PYTHONPATH so `python -m src.<package>` resolves;
# - defines the default local data layout described in the README.
#
# Every default below can be overridden from the environment, e.g.
#   DATA_ROOT=/path/to/LUNA16_preprocessed bash bash/luna16_volume_3d/run_resnet18_10fold.sh

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

activate_venv() {
  local candidate
  for candidate in "${VENV_PATH:-}" myenv .venv venv; do
    if [[ -n "${candidate}" && -f "${candidate}/bin/activate" ]]; then
      # shellcheck disable=SC1091
      source "${candidate}/bin/activate"
      return 0
    fi
  done
  echo "Warning: no virtual environment found (set VENV_PATH); using $(command -v python3)." >&2
}

if [[ "${SKIP_VENV:-0}" != "1" ]]; then
  activate_venv
fi

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Default data layout (see README.md, "Data").
DATA_ROOT="${DATA_ROOT:-data/processed}"
SPLITS_DIR="${SPLITS_DIR:-${DATA_ROOT}/cv_splits}"
SYNTHETIC_ROOT="${SYNTHETIC_ROOT:-data/synthetic_2d}"
DETECTOR_PREDICTIONS="${DETECTOR_PREDICTIONS:-outputs/cpmnetv2_luna16_10fold_bf16_guarded_results/normalized_predictions}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
