#!/bin/bash

# ================================
# Script per lanciare un esperimento Python in un virtualenv
# ================================

cd /home/domenico/lung_ct_3d2d_synthesis

# Nome dell'ambiente virtuale
VENV_NAME="venv"

# Script Python da eseguire
PYTHON_SCRIPT1="src/saliency.py"

# ================================
# Attivazione dell'ambiente virtuale
# ================================
echo "Attivazione dell'ambiente virtuale"
source $VENV_NAME/bin/activate

# ================================
# Lancio dello script Python
# ================================
echo "Esecuzione dello script: $PYTHON_SCRIPT1"
python $PYTHON_SCRIPT1

# ================================
# Disattivazione dell'ambiente virtuale
# ================================
deactivate
echo "Esperimento completato. Log salvato in $LOG_FILE"