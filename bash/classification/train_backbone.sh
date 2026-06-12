#!/bin/bash

export WANDB_API_KEY="5eb716fd87389d240533319f8751488e37103d23"

# ================================
# Script per lanciare un esperimento Python in un virtualenv
# ================================

cd /home/domenico/lung_ct_3d2d_synthesis

# Nome dell'ambiente virtuale
VENV_NAME="myenv"

# Script Python da eseguire
PYTHON_SCRIPT="src/train.py"

# ================================
# Attivazione dell'ambiente virtuale
# ================================
echo "Attivazione dell'ambiente virtuale"
source $VENV_NAME/bin/activate

# ================================
# Lancio dello script Python
# ================================
echo "Esecuzione dello script: $PYTHON_SCRIPT"
python $PYTHON_SCRIPT --config-name=config_2d

# ================================
# Disattivazione dell'ambiente virtuale
# ================================
deactivate
echo "Esperimento completato. Log salvato in $LOG_FILE"