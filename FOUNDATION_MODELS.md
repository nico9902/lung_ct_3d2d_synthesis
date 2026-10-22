# Foundation-model setup

Run these commands from the repository root. External code and pretrained
weights stay under the ignored `external/` directory. Record upstream commit
IDs and weight revisions alongside experiment configurations for repeat runs.

## 3DINO

Use the official [3DINO code](https://github.com/AICONSlab/3DINO) and
[teacher weights](https://huggingface.co/AICONSlab/3DINO-ViT).
Install the main `requirements.txt`, then obtain the external files:

```bash
mkdir -p external
git clone https://github.com/AICONSlab/3DINO.git external/3DINO
python -m pip install huggingface_hub
python -c 'from huggingface_hub import snapshot_download; snapshot_download("AICONSlab/3DINO-ViT", local_dir="external/3DINO-ViT")'
CHECKPOINT_PATH="$PWD/external/3DINO-ViT/3dino_vit_weights.pth" \
  bash bash/luna16_3dino_3d/run_3dino_baseline.sh
```

`DINO3D_CODE_DIR` overrides the source checkout (default: this repository's
`external/3DINO`). Direct Python entry points also accept `--code-dir`.
`CHECKPOINT_PATH` must point to the downloaded teacher `.pth` file.
The wrapper disables xFormers and uses PyTorch attention. The original
upstream training environment is not required by this backbone-only wrapper.
See [frozen probes](src/luna16_3dino_3d/README_frozen_probe.md) for the two
additional frozen-encoder experiments.

## Rad-JEPA-3D

Use the official [RadJepa code](https://github.com/huyquoctrinh/RadJepa) and
[Rad-JEPA-3D weights](https://huggingface.co/huyquoctrinh/Rad-Jepa-3D).
Feature extraction uses a separate environment for the Mamba CUDA extensions;
linear probes use the main repository environment. The launcher targets
PyTorch 2.5.1 with CUDA 12.1 for extraction.

```bash
mkdir -p external
git clone https://github.com/huyquoctrinh/RadJepa.git external/RadJepa
python3.10 -m venv external/radjepa-env
external/radjepa-env/bin/python -m pip install --upgrade pip setuptools wheel packaging ninja
external/radjepa-env/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
external/radjepa-env/bin/python -m pip install numpy pandas scipy scikit-learn SimpleITK einops huggingface_hub
external/radjepa-env/bin/python -m pip install causal-conv1d mamba-ssm --no-build-isolation
external/radjepa-env/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("huyquoctrinh/Rad-Jepa-3D", local_dir="external/Rad-Jepa-3D-hf")'
RADJEPA_PYTHON="$PWD/external/radjepa-env/bin/python" \
  bash bash/luna16_radjepa_3d/run_radjepa_baseline.sh
```

Building the CUDA extensions may require a local CUDA toolkit compatible with
PyTorch. Their wheels must match Python, PyTorch and CUDA; consult the
[Mamba installation instructions](https://github.com/state-spaces/mamba#installation)
if a source build fails. This setup recipe has not been reinstalled on a clean
CUDA machine as part of this repository cleanup.

`RADJEPA_CODE_DIR` must contain `vjepa_config.py` and `vjepa_model.py`.
`RADJEPA_HF_MODEL_DIR` must contain `config.json` and `encoder/encoder.pt`.
Defaults are `external/RadJepa` and `external/Rad-Jepa-3D-hf`.
Set `RAW_ROOT` if the raw `.mhd` scans are outside `data/raw/LUNA16/subsets`.
The launcher extracts four representations, trains ten probes per
representation, and aggregates the resulting predictions.
