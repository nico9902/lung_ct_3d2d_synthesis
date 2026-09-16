#!/bin/bash

# ==========================================================
# Reorganize preprocessed LUNA16 outputs into official folds
# ==========================================================

ORIGINAL_ROOT="/ssd2/domenico/datasets/LUNA16/subsets"

PREPROCESSED_ROOT="/ssd2/domenico/datasets/LUNA16_preprocessed"

echo "Reorganizing preprocessed scans into subset folders..."

# Loop over all subset folders
for subset_path in ${ORIGINAL_ROOT}/subset*
do
    subset_name=$(basename ${subset_path})

    echo "Processing ${subset_name}"

    mkdir -p "${PREPROCESSED_ROOT}/${subset_name}"

    # Find all original mhd scans
    find "${subset_path}" -name "*.mhd" | while read mhd_file
    do
        seriesuid=$(basename "${mhd_file}" .mhd)

        src_dir="${PREPROCESSED_ROOT}/${seriesuid}"
        dst_dir="${PREPROCESSED_ROOT}/${subset_name}/${seriesuid}"

        if [ -d "${src_dir}" ]; then
            echo "Moving ${seriesuid} -> ${subset_name}"
            mv "${src_dir}" "${dst_dir}"
        fi
    done
done

echo "=========================================="
echo "Reorganization completed"
echo "=========================================="