#!/bin/bash

# ==============================================================================
# GravitySpace Detection Training - Subvolume Memory Optimized
# ==============================================================================

PROJECT_DIR="/home/domenico/lung_ct_3d2d_synthesis"
cd "$PROJECT_DIR" || exit 1

VENV_PATH="myenv"
PYTHON_SCRIPT="src/det/GravitySpace/train.py"

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

echo "Executing subvolume training: $PYTHON_SCRIPT"

python "$PYTHON_SCRIPT" \
    exp.name=gravity_space_effnetV2s_grid10_subvolumes \
    model.backbone=EfficientNetV2-S \
    model.anchor_config=grid-10 \
    model.window_size=5 \
    model.sampling=1 \
    model.hidden_dim=128 \
    model.lr=1e-4 \
    model.loss.alpha=0.25 \
    model.loss.gamma=2.0 \
    model.loss.hook=10 \
    model.loss.hook_gap=5 \
    inference.save_qualitative=false \
    inference.qualitative_max_images=100 \
    inference.score_threshold=0.05 \
    inference.froc_normalization=scan \
    inference.distance=8 \
    inference.nms_box_radius=8 \
    inference.nms_z_radius=5 \
    inference.nms_2d_iou_threshold=0.3 \
    inference.nms_3d_iou_threshold=0.3 \
    model.ckpt="/home/domenico/lung_ct_3d2d_synthesis/outputs/gravity_space_effnetV2s_grid10_subvolumes/epoch\=11-step\=1500.ckpt" \
    data.batch_size=2 \
    data.input_mode=2d \
    data.context_slices=3 \
    data.use_subvolumes=true \
    data.subvolume_depth=32 \
    data.subvolume_stride=16 \
    data.val_subvolume_stride=32 \
    data.test_subvolume_stride=32 \
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
    trainer.accumulate_grad_batches=8 \
    trainer.devices="[0]" \
    trainer.precision=bf16-mixed \
    trainer.limit_test_batches=1.0 \

deactivate
echo "Task completed."
