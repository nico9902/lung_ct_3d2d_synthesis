#!/bin/bash

# ================================
# Script per lanciare un esperimento Python in un virtualenv
# ================================

cd /home/domenico/lung_ct_3d2d_synthesis

# Nome dell'ambiente virtuale
VENV_NAME="myenv"

# Script Python da eseguire
PYTHON_SCRIPT1="preprocess_dataset.py"
PYTHON_SCRIPT2="preprocess.py"

# ================================
# Attivazione dell'ambiente virtuale
# ================================
echo "Attivazione dell'ambiente virtuale"
source $VENV_NAME/bin/activate

# ================================
# Lancio dello script Python
# ================================
echo "Esecuzione dello script: $PYTHON_SCRIPT1"
python $PYTHON_SCRIPT1 --data_dir="/ssd2/Cantone/datasets/LIDC-IDRI/manifest-1600709154662/LIDC-IDRI" --output_dir="data/preprocessed_z_only" --resample_z_only

# ================================
# Lancio dello script Python
# ================================
echo "Esecuzione dello script: $PYTHON_SCRIPT2"
python $PYTHON_SCRIPT2 --data_dir="data/preprocessed_z_only" --output_dir="data/preprocessed_z_only"

# ================================
# Disattivazione dell'ambiente virtuale
# ================================
deactivate
echo "Esperimento completato. Log salvato in $LOG_FILE"