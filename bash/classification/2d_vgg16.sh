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
python $PYTHON_SCRIPT \
    exp.name=2d_vgg16_bs_16_512_512 \
    data.dataset_type=2d \
    data.dicom=false \
    data.csv_file=data/dataset_nodule_mean.csv \
    data.processed_dir=data/Lung2Dsynt_gt_nodules \
    data.batch_size=16 \
    data.num_workers=16 \
    data.train_transforms.transforms.0._target_=torchvision.transforms.Resize \
    "data.train_transforms.transforms.0.size=[512, 512]" \
    data.train_transforms.transforms.1._target_=torchvision.transforms.ToTensor \
    "data.train_transforms.transforms.2._target_=torchvision.transforms.Normalize" \
    "data.train_transforms.transforms.2.mean=[0.485, 0.456, 0.406]" \
    "data.train_transforms.transforms.2.std=[0.229, 0.224, 0.225]" \
    data.test_transforms.transforms.0._target_=torchvision.transforms.Resize \
    "data.test_transforms.transforms.0.size=[512, 512]" \
    data.test_transforms.transforms.1._target_=torchvision.transforms.ToTensor \
    "data.test_transforms.transforms.2._target_=torchvision.transforms.Normalize" \
    "data.test_transforms.transforms.2.mean=[0.485, 0.456, 0.406]" \
    "data.test_transforms.transforms.2.std=[0.229, 0.224, 0.225]" \
    model.network_name=BackboneClassifier2D \
    model.backbone_name=vgg16 \
    model.num_classes=2 \
    model.freeze_half=false \
    model.lr=1e-4 \
    trainer.max_epochs=50 \
    trainer.accumulate_grad_batches=1 \
    trainer.precision=16-mixed \
    "trainer.devices=[3]"

# ================================
# Disattivazione dell'ambiente virtuale
# ================================
deactivate
echo "Esperimento completato. Log salvato in $LOG_FILE"