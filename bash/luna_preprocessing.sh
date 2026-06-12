#!/bin/bash

cd /home/domenico/lung_ct_3d2d_synthesis || exit 1

VENV_NAME="myenv"
PYTHON_SCRIPT="src/det/SCPMNet/luna_preprocessing.py"

LUNA_ROOT="/ssd2/domenico/datasets/LUNA16"
OUTPUT_DIR="/ssd2/domenico/datasets/LUNA16_preprocessed"
LABELS_CSV="${OUTPUT_DIR}/luna16_labels.csv"

echo "Attivazione dell'ambiente virtuale"
source "${VENV_NAME}/bin/activate"

echo "Controllo CUDA/PyTorch"
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo "Esecuzione dello script: ${PYTHON_SCRIPT}"
python "${PYTHON_SCRIPT}" \
  --dicom-dir "${LUNA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --labels-csv "${LABELS_CSV}" \
  --spacing 1 1 1 \
  --num-workers 1

deactivate
echo "Preprocessing completato"