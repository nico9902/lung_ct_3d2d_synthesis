#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"

VENV_PATH="${VENV_PATH:-myenv}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
  source "${VENV_PATH}/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

LUNA16_PROCESSED_DIR="${LUNA16_PROCESSED_DIR:-/ssd2/domenico/datasets/LUNA16_preprocessed}"
SPLITS_DIR="${SPLITS_DIR:-${LUNA16_PROCESSED_DIR}/cv_splits}"
FOLDS="${FOLDS:-0}"
CSV_FILE_PREFIX="${CSV_FILE_PREFIX:-${SPLITS_DIR}/luna16_classification_fold}"
CSV_FILE_SUFFIX="${CSV_FILE_SUFFIX:-.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/luna16_saliency_synthetic}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_DIR}/src/luna16_synthetic_2d/conf}"
ONLY_NO_NODULES="${ONLY_NO_NODULES:-false}"
RBF_SMOOTH="${RBF_SMOOTH:-0.1}"
NUM_BOUNDARY_ANCHORS="${NUM_BOUNDARY_ANCHORS:-24}"
USE_LUNG_VOLUME_ANCHORS="${USE_LUNG_VOLUME_ANCHORS:-true}"
LUNG_ANCHOR_ERODE_ITERATIONS="${LUNG_ANCHOR_ERODE_ITERATIONS:-1}"
SNAP_SURFACE_TO_LUNG="${SNAP_SURFACE_TO_LUNG:-false}"
ANCHOR_MIN_LUNG_AREA_FRACTION="${ANCHOR_MIN_LUNG_AREA_FRACTION:-0.35}"
NUM_CONTOUR_POINTS="${NUM_CONTOUR_POINTS:-4}"
USE_SAVED_LUNG_MASKS="${USE_SAVED_LUNG_MASKS:-true}"
LUNG_MASK_METHOD="${LUNG_MASK_METHOD:-body_threshold}"
NORMALIZED_AIR_THRESHOLD="${NORMALIZED_AIR_THRESHOLD:-0.35}"
HU_AIR_MIN="${HU_AIR_MIN:--1000}"
HU_AIR_MAX="${HU_AIR_MAX:--320}"
BODY_THRESHOLD_PERCENTILE="${BODY_THRESHOLD_PERCENTILE:-1}"
LUNG_COMPONENT_COUNT="${LUNG_COMPONENT_COUNT:-2}"
LUNGMASK_MODEL_NAME="${LUNGMASK_MODEL_NAME:-R231}"
LUNGMASK_FORCE_CPU="${LUNGMASK_FORCE_CPU:-false}"
PSEUDO_MIN_REGIONS="${PSEUDO_MIN_REGIONS:-1}"
PSEUDO_MAX_REGIONS="${PSEUDO_MAX_REGIONS:-4}"
PSEUDO_MIN_RADIUS="${PSEUDO_MIN_RADIUS:-8}"
PSEUDO_MAX_RADIUS="${PSEUDO_MAX_RADIUS:-20}"
PSEUDO_ERODE_ITERATIONS="${PSEUDO_ERODE_ITERATIONS:-2}"
PSEUDO_CENTRAL_PERCENTILE="${PSEUDO_CENTRAL_PERCENTILE:-70}"
PSEUDO_MIN_SLICE_AREA_PERCENTILE="${PSEUDO_MIN_SLICE_AREA_PERCENTILE:-35}"
PSEUDO_EMPIRICAL_POSITION_ATTEMPTS="${PSEUDO_EMPIRICAL_POSITION_ATTEMPTS:-100}"
PSEUDO_MAX_ATTEMPTS="${PSEUDO_MAX_ATTEMPTS:-5}"
MIN_LUNG_COVERAGE="${MIN_LUNG_COVERAGE:-0.25}"
MIN_BEST_LUNG_COVERAGE="${MIN_BEST_LUNG_COVERAGE:-0.10}"
USE_EMPIRICAL_PSEUDO_NODULES="${USE_EMPIRICAL_PSEUDO_NODULES:-true}"
EMPIRICAL_NODULE_DISTRIBUTION_PATH="${EMPIRICAL_NODULE_DISTRIBUTION_PATH:-outputs/luna16_saliency_control_point_distribution/empirical_nodule_distribution_from_control_points.npz}"
LUNG_WINDOW_CENTER="${LUNG_WINDOW_CENTER:--600}"
LUNG_WINDOW_WIDTH="${LUNG_WINDOW_WIDTH:-1500}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

for FOLD in ${FOLDS}; do
  CSV_FILE="${CSV_FILE_PREFIX}${FOLD}${CSV_FILE_SUFFIX}"
  OUTPUT_DIR="${OUTPUT_ROOT}"
  HYDRA_OUTPUT_DIR="${OUTPUT_ROOT}/hydra/fold${FOLD}"

  if [[ "${CSV_FILE}" == *"{"* || "${CSV_FILE}" == *"}"* ]]; then
    echo "Error: CSV path still contains a brace placeholder: ${CSV_FILE}" >&2
    echo "Set CSV_FILE_PREFIX and CSV_FILE_SUFFIX instead of using a {fold} template." >&2
    exit 1
  fi

  echo "Generating LUNA16 saliency surfaces for fold ${FOLD}"
  echo "  CSV: ${CSV_FILE}"
  echo "  Output: ${OUTPUT_DIR}"

  python3 src/luna16_synthetic_2d/saliency.py \
    --config-path "${CONFIG_DIR}" \
    --config-name config \
    data.dataset_type=luna16_nii \
    data.processed_dir="${LUNA16_PROCESSED_DIR}" \
    data.csv_file="${CSV_FILE}" \
    data.num_workers=0 \
    data.return_lung_mask=true \
    data.only_no_nodules="${ONLY_NO_NODULES}" \
    saliency.save_path="${OUTPUT_DIR}" \
    saliency.debug=false \
    saliency.show_plot=false \
    saliency.save_surface_grid=true \
    saliency.rbf_smooth="${RBF_SMOOTH}" \
    saliency.num_boundary_anchors="${NUM_BOUNDARY_ANCHORS}" \
    saliency.use_lung_volume_anchors="${USE_LUNG_VOLUME_ANCHORS}" \
    saliency.lung_anchor_erode_iterations="${LUNG_ANCHOR_ERODE_ITERATIONS}" \
    saliency.snap_surface_to_lung="${SNAP_SURFACE_TO_LUNG}" \
    saliency.anchor_min_lung_area_fraction="${ANCHOR_MIN_LUNG_AREA_FRACTION}" \
    saliency.num_contour_points="${NUM_CONTOUR_POINTS}" \
    saliency.use_saved_lung_masks="${USE_SAVED_LUNG_MASKS}" \
    saliency.lung_mask_method="${LUNG_MASK_METHOD}" \
    saliency.normalized_air_threshold="${NORMALIZED_AIR_THRESHOLD}" \
    saliency.hu_air_min="${HU_AIR_MIN}" \
    saliency.hu_air_max="${HU_AIR_MAX}" \
    saliency.body_threshold_percentile="${BODY_THRESHOLD_PERCENTILE}" \
    saliency.lung_component_count="${LUNG_COMPONENT_COUNT}" \
    saliency.lungmask_model_name="${LUNGMASK_MODEL_NAME}" \
    saliency.lungmask_force_cpu="${LUNGMASK_FORCE_CPU}" \
    saliency.pseudo_min_regions="${PSEUDO_MIN_REGIONS}" \
    saliency.pseudo_max_regions="${PSEUDO_MAX_REGIONS}" \
    saliency.pseudo_min_radius="${PSEUDO_MIN_RADIUS}" \
    saliency.pseudo_max_radius="${PSEUDO_MAX_RADIUS}" \
    saliency.pseudo_erode_iterations="${PSEUDO_ERODE_ITERATIONS}" \
    saliency.pseudo_central_percentile="${PSEUDO_CENTRAL_PERCENTILE}" \
    saliency.pseudo_min_slice_area_percentile="${PSEUDO_MIN_SLICE_AREA_PERCENTILE}" \
    saliency.pseudo_empirical_position_attempts="${PSEUDO_EMPIRICAL_POSITION_ATTEMPTS}" \
    saliency.pseudo_max_attempts="${PSEUDO_MAX_ATTEMPTS}" \
    saliency.min_lung_coverage="${MIN_LUNG_COVERAGE}" \
    saliency.min_best_lung_coverage="${MIN_BEST_LUNG_COVERAGE}" \
    saliency.use_empirical_pseudo_nodules="${USE_EMPIRICAL_PSEUDO_NODULES}" \
    saliency.empirical_nodule_distribution_path="${EMPIRICAL_NODULE_DISTRIBUTION_PATH}" \
    saliency.lung_window_center="${LUNG_WINDOW_CENTER}" \
    saliency.lung_window_width="${LUNG_WINDOW_WIDTH}" \
    hydra.run.dir="${HYDRA_OUTPUT_DIR}"
done
