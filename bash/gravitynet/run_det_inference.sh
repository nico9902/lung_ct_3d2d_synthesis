#!/bin/bash

# ==============================================================================
# GravitySpace Detection Training Script
# ==============================================================================

# Project root directory
PROJECT_DIR="/home/domenico/lung_ct_3d2d_synthesis"
cd "$PROJECT_DIR" || exit 1

# # Optional: WandB API Key
# export WANDB_API_KEY="5eb716fd87389d240533319f8751488e37103d23"

# Virtual environment path
VENV_PATH="myenv"

# Python script path
PYTHON_SCRIPT="src/det/GravitySpace/detections_test_distance.py"

# ==============================================================================
# Activate Virtual Environment
# ==============================================================================
echo "Activating virtual environment..."
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
elif [ -f "venv/bin/activate" ]; then
    echo "Warning: '$VENV_PATH' not found, falling back to 'venv'..."
    source venv/bin/activate
else
    echo "Error: Virtual environment not found."
    exit 1
fi

# ==============================================================================
# Run Training with Hydra overrides
# ==============================================================================
echo "Executing: $PYTHON_SCRIPT"

# You can adjust these parameters as needed
python "$PYTHON_SCRIPT" \
    model.ckpt="/home/domenico/lung_ct_3d2d_synthesis/outputs/gravity_space_effnetv2s_grid10_subvolumes/epoch\=12-step\=1625.ckpt" \
    exp.name=gravity_space_effnetv2s_grid10_subvolumes \
    model.backbone=EfficientNet-V2-S \
    model.anchor_config=grid-10 \
    model.window_size=5 \
    model.sampling=1 \
    model.hidden_dim=128 \
    model.lr=1e-4 \
    model.loss.alpha=0.25 \
    model.loss.gamma=2.0 \
    model.loss.hook=10 \
    model.loss.hook_gap=5 \
    data.batch_size=1 \
    data.image_size=[448,320] \
    data.csv_path="data/dataset_nodule_mean_all.csv" \
    data.images_dir="/ssd2/domenico/datasets/LIDC-IDRI_nifti_z_only_processed" \
    data.annotations_dir="/ssd2/domenico/datasets/LIDC-IDRI_nifti_z_only_processed" \
    data.precomputed_slices_dir="/ssd2/domenico/datasets/LIDC-IDRI_nifti_z_only_processed" \
    data.precomputed_centers_dir="/ssd2/domenico/datasets/LIDC-IDRI_nifti_z_only_processed" \
    trainer.max_epochs=50 \
    trainer.accumulate_grad_batches=4 \
    trainer.devices="[3]" \
    trainer.precision=bf16-mixed \
    inference.save_qualitative=true \
    inference.qualitative_max_images=100 \
    inference.score_threshold=0.05 \

# ==============================================================================
# Deactivate
# ==============================================================================
deactivate
echo "Task completed."
