#!/bin/bash

# ================================
# Script per lanciare un esperimento Python in un virtualenv
# ================================

cd /home/domenico/lung_ct_3d2d_synthesis

# Nome dell'ambiente virtuale
VENV_NAME="myenv"

# Script Python da eseguire
PYTHON_SCRIPT="src/det/SCPMNet/lidc_preprocessing.py"

# ================================
# Attivazione dell'ambiente virtuale
# ================================
echo "Attivazione dell'ambiente virtuale"
source $VENV_NAME/bin/activate

# ================================
# Lancio dello script Python
# ================================
echo "Esecuzione dello script: $PYTHON_SCRIPT"
python $PYTHON_SCRIPT --dicom-dir="/ssd2/Cantone/datasets/LIDC-IDRI/manifest-1600709154662/LIDC-IDRI" --output-dir="/ssd2/domenico/datasets/lidc_process" --num-workers=8 --labels-only --labels-csv="/ssd2/domenico/datasets/lidc_process/lidc_labels.csv"

# ================================
# Disattivazione dell'ambiente virtuale
# ================================
deactivate
echo "Esperimento completato. Log salvato in $LOG_FILE"
