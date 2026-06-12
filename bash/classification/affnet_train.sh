#!/bin/bash

# ================================
# Training AFFNet with Multi-View Loss
# ================================

# Set WANDB API KEY if needed
# export WANDB_API_KEY="YOUR_KEY_HERE"

# Current project directory
PROJECT_DIR=$(pwd)
cd $PROJECT_DIR

# Virtual environment name (defaulting to 'venv' as seen in the root, or 'myenv' as in previous scripts)
VENV_NAME="venv"

# Script Python mapping
PYTHON_SCRIPT="src/train.py"

# ================================
# Activate Virtual Environment
# ================================
if [ -d "$VENV_NAME" ]; then
    echo "Activating virtual environment: $VENV_NAME"
    source $VENV_NAME/bin/activate
elif [ -d "myenv" ]; then
    echo "Activating virtual environment: myenv"
    source myenv/bin/activate
else
    echo "Warning: No virtual environment found at $VENV_NAME or myenv"
fi

# ================================
# Run Training
# ================================
echo "Running AFFNet experiment..."

python $PYTHON_SCRIPT \
    exp.name=affnet_multiview_experiment \
    data.dataset_type=3d \
    data.dicom=false \
    data.csv_file=data/dataset_nodule_mean.csv \
    data.processed_dir=data/preprocessed \
    data.return_mask=false \
    data.batch_size=1 \
    data.num_workers=8 \
    model.network_name=AFFNet \
    model.num_classes=2 \
    model.loss.name=AFFNetLoss \
    #model.loss.weight2=0.4 \
    #model.loss.weight_common=0.4 \
    model.lr=1e-4 \
    trainer.max_epochs=50 \
    trainer.accumulate_grad_batches=16 \
    trainer.precision=16-mixed \
    "trainer.devices=[0]"

# ================================
# Deactivate
# ================================
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

echo "Experiment scheduled/completed!"
