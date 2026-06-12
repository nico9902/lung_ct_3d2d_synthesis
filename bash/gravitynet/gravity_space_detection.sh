#!/bin/bash

# ==============================================================================
# GravitySpace Detection Training Script
# ==============================================================================

# Project root directory
PROJECT_DIR="/home/domenico/lung_ct_3d2d_synthesis"
cd "$PROJECT_DIR" || exit 1

# Optional: WandB API Key
export WANDB_API_KEY="5eb716fd87389d240533319f8751488e37103d23"

# Virtual environment path
VENV_PATH="myenv"

# Python script path
PYTHON_SCRIPT="src/det/GravitySpace/train.py"

# ==============================================================================
# GPU DEVICE MAPPING (Important!)
# ==============================================================================
# On this system, PyTorch device IDs DO NOT match nvidia-smi GPU IDs
# Mapping discovered via test_gpu_mapping.py:
#   trainer.devices=[0] → GPU 3 (A100 80GB) - AVOID: Out of memory
#   trainer.devices=[1] → GPU 0 (V100 16GB) ✓ RECOMMENDED
#   trainer.devices=[2] → GPU 1 (V100 16GB) ✓ Alternative
#   trainer.devices=[3] → GPU 2 (V100 16GB) ✓ Alternative
#
# Current selection: trainer.devices=[1] (GPU 0 in nvidia-smi)
# To verify: watch -n 2 nvidia-smi
# ==============================================================================

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

# ==============================================================================
# Parameter Guidelines
# ==============================================================================
# V100 (16GB) - Memory Optimized:
#   model.chunk_size=50     (process 50 slices at a time, reduces peak memory ~65%)
#   model.hidden_dim=64
#   model.window_size=3
#   data.image_size=[448,320]
#   data.use_subvolumes=true
#   data.subvolume_depth=32
#   data.subvolume_stride=16
#   data.val_subvolume_stride=16
#   data.test_subvolume_stride=16
#   data.positive_fraction=0.7
#
# A100 (80GB) - Speed Optimized (disable chunking):
#   model.chunk_size=null   (process all slices at once, faster but needs more memory)
#   model.hidden_dim=256    (can increase for better model capacity)
#   model.window_size=7     (can increase for larger receptive field)
#   data.image_size=[512,512]
# ==============================================================================

# You can adjust these parameters as needed
python "$PYTHON_SCRIPT" \
    exp.name=gravity_space_res18_grid10 \
    model.backbone=ResNet-18 \
    model.anchor_config=grid-10 \
    model.window_size=5 \
    model.sampling=1 \
    model.hidden_dim=64 \
    model.lr=1e-4 \
    model.loss.alpha=0.25 \
    model.loss.gamma=2.0 \
    model.loss.hook=10 \
    model.loss.hook_gap=5 \
    data.batch_size=2 \
    data.use_subvolumes=true \
    data.subvolume_depth=32 \
    data.subvolume_stride=16 \
    data.val_subvolume_stride=16 \
    data.test_subvolume_stride=16 \
    data.positive_fraction=0.7 \
    data.samples_per_epoch=2000 \
    data.image_size=[448,320] \
    data.csv_path="data/dataset_nodule_mean.csv" \
    data.images_dir="/ssd2/domenico/datasets/LIDC-IDRI_nifti_z_only_processed" \
    data.annotations_dir="/ssd2/domenico/datasets/LIDC-IDRI_nifti_z_only_processed" \
    data.precomputed_slices_dir="/ssd2/domenico/datasets/LIDC-IDRI_nifti_z_only_processed" \
    data.precomputed_centers_dir="/ssd2/domenico/datasets/LIDC-IDRI_nifti_z_only_processed" \
    data.num_workers=8 \
    trainer.max_epochs=50 \
    trainer.accumulate_grad_batches=4 \
    trainer.devices="[0]" \
    trainer.precision=bf16-mixed

# ==============================================================================
# Deactivate
# ==============================================================================
deactivate
echo "Task completed."
