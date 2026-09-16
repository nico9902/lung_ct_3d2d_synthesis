#!/bin/bash

# ================================
# Script for DICOM to NIfTI Batch Conversion (including .npy support)
# ================================

# Project root directory
PROJECT_DIR="/Users/domenicopaolo/Documents/works/lung_ct_3d2d_synthesis"
cd "$PROJECT_DIR"

# Virtual environment name (using the one detected in the root)
VENV_PATH="myenv"

# Script details
PYTHON_SCRIPT="dicom_to_nifti.py"

# Default paths (Adjust as needed)
INPUT_DIR="data/preprocessed_z_only"
OUTPUT_DIR="/ssd2/domenico/datasets/LIDC-IDRI_nifti_z_only"

# ================================
# Activate virtual environment
# ================================
echo "Activating virtual environment..."
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# ================================
# Run Python script
# ================================
# You can uncomment and modify these lines depending on your use case:

# SCENARIO 1: Batch convert preprocessed .npy volumes and masks (resized to 480x352)
echo "Running batch NPY to NIfTI conversion from: $INPUT_DIR"
python "$PYTHON_SCRIPT" "$INPUT_DIR" "$OUTPUT_DIR" --batch --npy

# SCENARIO 2: Batch convert raw DICOM folders
# echo "Running batch DICOM to NIfTI conversion..."
# python "$PYTHON_SCRIPT" "path/to/dicom_root" "path/to/output_nifti" --batch

# ================================
# Deactivate virtual environment
# ================================
deactivate
echo "Conversion task completed."
