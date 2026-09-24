#!/usr/bin/env bash
# Move preprocessed LUNA16 scan folders into the official subset0..subset9
# layout of the raw dataset (<DATA_ROOT>/<subset>/<seriesuid>/).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

LUNA16_SUBSETS="${LUNA16_SUBSETS:-data/raw/LUNA16/subsets}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${DATA_ROOT}}"

echo "Reorganizing preprocessed scans into subset folders..."

for subset_path in "${LUNA16_SUBSETS}"/subset*; do
  subset_name="$(basename "${subset_path}")"
  echo "Processing ${subset_name}"
  mkdir -p "${PREPROCESSED_ROOT}/${subset_name}"

  find "${subset_path}" -name "*.mhd" | while read -r mhd_file; do
    seriesuid="$(basename "${mhd_file}" .mhd)"
    src_dir="${PREPROCESSED_ROOT}/${seriesuid}"
    dst_dir="${PREPROCESSED_ROOT}/${subset_name}/${seriesuid}"

    if [[ -d "${src_dir}" ]]; then
      echo "Moving ${seriesuid} -> ${subset_name}"
      mv "${src_dir}" "${dst_dir}"
    fi
  done
done

echo "Reorganization completed"
