#!/bin/bash

# ================================
# Script per lanciare un esperimento Python in un virtualenv
# ================================

cd /home/domenico/lung_ct_3d2d_synthesis

# Nome dell'ambiente virtuale
VENV_NAME="myenv"

# Script Python da eseguire
PYTHON_SCRIPT="generate_slice_labels.py"

# ================================
# Attivazione dell'ambiente virtuale
# ================================
echo "Attivazione dell'ambiente virtuale"
source $VENV_NAME/bin/activate

# ================================
# Lancio dello script Python
# ================================
echo "Esecuzione dello script: $PYTHON_SCRIPT"
python $PYTHON_SCRIPT 

# ================================
# Disattivazione dell'ambiente virtuale
# ================================
deactivate
echo "Esperimento completato. Log salvato in $LOG_FILE"