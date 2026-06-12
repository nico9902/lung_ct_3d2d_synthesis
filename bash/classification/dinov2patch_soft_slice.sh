#!/bin/bash

export WANDB_API_KEY="5eb716fd87389d240533319f8751488e37103d23"

# ================================
# Script per lanciare un esperimento Python in un virtualenv
# ================================

cd /Users/domenicopaolo/Documents/works/lung_ct_3d2d_synthesis

# Nome dell'ambiente virtuale
VENV_NAME="myenv"

# Script Python da eseguire
PYTHON_SCRIPT="src/train.py"

# ================================
# Attivazione dell'ambiente virtuale
# ================================
echo "Attivazione dell'ambiente virtuale"
# Adjust path if needed for local mac environment
if [ -d "$VENV_NAME" ]; then
    source $VENV_NAME/bin/activate
else
    echo "Warning: $VENV_NAME not found. Trying to run with system python."
fi

# ================================
# Lancio dello script Python
# ================================
echo "Esecuzione dello script: $PYTHON_SCRIPT"
python $PYTHON_SCRIPT \
    exp.name=dinov2_patch_soft_slice_mip_10_5 \
    data.dataset_type=3d \
    data.dicom=false \
    data.csv_file=data/dataset_nodule_mean.csv \
    data.processed_dir=data/preprocessed_z_only \
    data.return_mask=false \
    data.batch_size=1 \
    data.num_workers=8 \
    model.network_name=DinoV2Patch_SoftSlice \
    model.num_classes=2 \
    model.patch_size=14 \
    "model.img_size=[504,504]" \
    model.lr=1e-4 \
    model.optimizer.name=AdamW \
    model.optimizer.weight_decay=1e-2 \
    model.scheduler.name=CosineAnnealingLR \
    model.scheduler.T_max=10 \
    model.scheduler.eta_min=1e-6 \
    "data.train_transforms.transforms=[{_target_: src.transforms.Custom_Resize, size: [392, 392]}, {_target_: src.transforms.CustomToTensor}, {_target_: src.transforms.RepeatChannels, repeats: 3}, {_target_: src.transforms.MIP, slab_size: 10, stride: 5}]" \
    "data.test_transforms.transforms=[{_target_: src.transforms.Custom_Resize, size: [392, 392]}, {_target_: src.transforms.CustomToTensor}, {_target_: src.transforms.RepeatChannels, repeats: 3}, {_target_: src.transforms.MIP, slab_size: 10, stride: 5}]" \
    trainer.max_epochs=50 \
    trainer.accumulate_grad_batches=16 \
    trainer.precision=16-mixed \
    "trainer.devices=[0]"

# ================================
# Disattivazione dell'ambiente virtuale
# ================================
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi
echo "Esperimento completato."
