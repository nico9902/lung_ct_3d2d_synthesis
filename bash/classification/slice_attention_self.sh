#!/bin/bash

export WANDB_API_KEY="5eb716fd87389d240533319f8751488e37103d23"

# ================================
# Training SliceAttentionNetwork with Self-Attention (CLS Token)
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
echo "Esecuzione dello script: $PYTHON_SCRIPT with self-attention"
python $PYTHON_SCRIPT \
    exp.name=slice_attention_vgg16_self_supervised_attn \
    data.dataset_type=3d \
    data.dicom=false \
    data.csv_file=data/dataset_nodule_mean.csv \
    data.processed_dir=data/processed \
    data.return_mask=true \
    data.batch_size=1 \
    data.num_workers=16 \
    model.network_name=SliceAttentionNetwork \
    model.backbone_name=vgg16 \
    model.num_classes=2 \
    model.feature_dim=null \
    model.freeze_half=true \
    model.enable_segmentation=false \
    model.segmentation_feature_dim=256 \
    model.segmentation_loss_weight=0.1 \
    model.attention_type=self_attention_cls \
    model.supervise_attention=true \
    model.attention_loss_weight=1.0 \
    model.lr=1e-4 \
    "data.train_transforms.transforms=[{_target_: src.transforms.Custom_Resize, size: [512, 512]}, {_target_: src.transforms.CustomToTensor}]" \
    "data.test_transforms.transforms=[{_target_: src.transforms.Custom_Resize, size: [512, 512]}, {_target_: src.transforms.CustomToTensor}]" \
    trainer.max_epochs=50 \
    trainer.accumulate_grad_batches=16 \
    trainer.precision=16-mixed \
    "trainer.devices=[0]"

# ================================
# Disattivazione dell'ambiente virtuale
# ================================
deactivate
echo "Esperimento completato!"
