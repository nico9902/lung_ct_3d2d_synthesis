#!/bin/bash

export WANDB_API_KEY="5eb716fd87389d240533319f8751488e37103d23"

# Nome dell'ambiente virtuale (modifica se necessario)
VENV_NAME="myenv"

# Script Python da eseguire
PYTHON_SCRIPT="src/train.py"

# Attivazione dell'ambiente virtuale
echo "Attivazione dell'ambiente virtuale"
source $VENV_NAME/bin/activate

# Lancio dello script Python con config_detect
echo "Esecuzione dello script: $PYTHON_SCRIPT con config_detect"
python $PYTHON_SCRIPT --config-name config_detect \
    exp.name=dino_detector \
    data.dataset_type=3d \
    data.dicom=false \
    data.csv_file=data/dataset_nodule_mean.csv \
    data.processed_dir=data/processed \
    data.return_mask=true \
    data.num_workers=16 \
    model.optimizer.name=AdamW \
    model.optimizer.weight_decay=1e-2 \
    model.scheduler.name=CosineAnnealingLR \
    model.scheduler.T_max=10 \
    model.scheduler.eta_min=1e-6 \
    "data.train_transforms.transforms=[{_target_: src.transforms.Custom_Resize, size: [504, 504]}, {_target_: src.transforms.CustomToTensor}]" \
    "data.test_transforms.transforms=[{_target_: src.transforms.Custom_Resize, size: [504, 504]}, {_target_: src.transforms.CustomToTensor}]" \
    trainer.max_epochs=50 \
    trainer.accumulate_grad_batches=16 \
    trainer.precision=16-mixed \
    trainer.devices=[0] \
    data.batch_size=1

# Disattivazione dell'ambiente virtuale
deactivate
echo "Esperimento completato."
