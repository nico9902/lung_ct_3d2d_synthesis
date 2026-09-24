"""3DINO-ViT backbone: official loading, partial-freeze logic, windowing, and preprocessing.

3DINO-ViT (Xu et al., npj Digital Medicine 2025, arXiv:2501.11755;
https://github.com/AICONSlab/3DINO) is a ViT-Large-3D pretrained with a
DINOv2-style joint-embedding objective on 3D medical volumes. The official
code is vendored (unmodified) under ``external/3DINO`` and imported here
rather than reimplemented, per the official implementation:

  - Architecture (verified in ``dinov2/models/vision_transformer.py``):
    ``vit_large_3d`` -> ``DinoVisionTransformer3d(embed_dim=1024, depth=24,
    num_heads=16, mlp_ratio=4, patch_size=16, in_chans=1)``, built with
    ``block_chunks=4`` (matching the released checkpoint's config), where
    ``self.blocks`` is 4 ``BlockChunk`` groups, each left-padded with
    ``nn.Identity()`` placeholders so a flat global block index is
    recoverable (see ``flatten_transformer_blocks`` below).
  - High-resolution config (``dinov2/configs/train/vit3d_highres.yaml``,
    used by the official ``notebooks/basic_model_use.ipynb`` demo):
    ``crops.global_crops_size: 112`` -> native input ``(1, 1, 112, 112, 112)``
    (112 / patch_size 16 = 7, so no padding is needed for a single window).
  - Global representation (verified against the official demo notebook,
    NOT assumed): the notebook calls the model directly, ``out = model(x)``
    -> shape ``(B, 1024)``. ``DinoVisionTransformer.forward`` (with the
    default ``is_training=False``) returns
    ``self.head(forward_features(x)["x_norm_clstoken"])``, and ``self.head``
    is ``nn.Identity()`` -- i.e. plain ``model(x)`` already returns the
    post-LayerNorm CLS token, exactly matching the notebook's output shape.
    We use this as the per-window embedding.
  - Preprocessing (verified against the official demo notebook, exact
    formula, not the undocumented ``ScaleIntensityRangePercentilesd``
    defaults from ``dinov2/data/transforms.py`` which has no default
    ``min_int``): per-volume/per-window percentile rescaling to ``[-1, 1]``::

        min_val = quantile(x, 0.0005); max_val = quantile(x, 0.9995)
        x = (x - min_val) / (max_val - min_val)
        x = clip(x * 2 - 1, -1, 1)

  - Checkpoint loading (verified in ``dinov2/utils/utils.py::load_pretrained_weights``):
    ``torch.load(path)["teacher"]``, strip ``"module."``/``"backbone."``
    prefixes, ``load_state_dict(strict=False)``. Imported directly, not
    reimplemented.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch
import torch.nn as nn

DEFAULT_3DINO_CODE_DIR = Path("/home/domenico/lung_ct_3d2d_synthesis/external/3DINO")
NATIVE_INPUT_SIZE = 112  # dinov2/configs/train/vit3d_highres.yaml: crops.global_crops_size
PATCH_SIZE = 16
EMBED_DIM = 1024
NUM_BLOCKS = 24
BLOCK_CHUNKS = 4  # student.block_chunks in dinov2/configs/ssl3d_default_config.yaml


def add_3dino_to_path(code_dir: Path = DEFAULT_3DINO_CODE_DIR) -> None:
    path_str = str(Path(code_dir).resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def build_backbone(code_dir: Path = DEFAULT_3DINO_CODE_DIR) -> nn.Module:
    """Build the un-pretrained ViT-Large-3D architecture (official factory function)."""
    import os

    os.environ.setdefault("XFORMERS_DISABLED", "1")  # xformers not installed; official code
    # falls back to plain PyTorch attention when unavailable (dinov2/layers/attention.py).
    add_3dino_to_path(code_dir)
    from dinov2.models.vision_transformer import vit_large_3d

    return vit_large_3d(
        patch_size=PATCH_SIZE,
        img_size=NATIVE_INPUT_SIZE,
        in_chans=1,
        block_chunks=BLOCK_CHUNKS,
        ffn_layer="mlp",
    )


def load_checkpoint(model: nn.Module, checkpoint_path: Path, code_dir: Path = DEFAULT_3DINO_CODE_DIR) -> None:
    """Official checkpoint loading: dinov2.utils.utils.load_pretrained_weights(model, path, "teacher")."""
    add_3dino_to_path(code_dir)
    from dinov2.utils.utils import load_pretrained_weights

    load_pretrained_weights(model, str(checkpoint_path), "teacher")


def flatten_transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Return the 24 real transformer Block modules in global order.

    With ``block_chunks=4`` the official code stores ``model.blocks`` as 4
    ``BlockChunk`` (``nn.ModuleList``) groups, each left-padded with
    ``nn.Identity()`` placeholders so that a flat index into the
    concatenation of all chunks matches the block's true global position
    (see ``DinoVisionTransformer.__init__`` in ``vision_transformer.py``).
    Filtering out the ``nn.Identity`` placeholders and concatenating the
    remaining modules across chunks, in order, recovers the 24 real blocks
    in the correct global order.
    """
    blocks: list[nn.Module] = []
    for chunk in model.blocks:
        blocks.extend(module for module in chunk if not isinstance(module, nn.Identity))
    if len(blocks) != NUM_BLOCKS:
        raise RuntimeError(f"Expected {NUM_BLOCKS} transformer blocks, found {len(blocks)}")
    return blocks


def apply_partial_freeze(model: nn.Module) -> dict[str, object]:
    """Freeze patch embedding/early input layers and the first half of the
    transformer blocks; keep the second half of the blocks and the final
    norm trainable, per the task specification::

        freeze blocks [0, ..., N/2 - 1]
        fine-tune blocks [N/2, ..., N - 1], final norm

    Returns a report dict with parameter counts and frozen/trainable module
    names, for printing and logging.
    """
    blocks = flatten_transformer_blocks(model)
    split = NUM_BLOCKS // 2

    frozen_modules = {"patch_embed": model.patch_embed}
    for name, module in frozen_modules.items():
        for param in module.parameters():
            param.requires_grad = False
    model.cls_token.requires_grad = False
    model.pos_embed.requires_grad = False
    model.mask_token.requires_grad = False

    frozen_block_names = []
    trainable_block_names = []
    for i, block in enumerate(blocks):
        trainable = i >= split
        for param in block.parameters():
            param.requires_grad = trainable
        (trainable_block_names if trainable else frozen_block_names).append(f"blocks.{i}")

    for param in model.norm.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "num_blocks": NUM_BLOCKS,
        "frozen_blocks": list(range(split)),
        "trainable_blocks": list(range(split, NUM_BLOCKS)),
        "frozen_module_names": ["patch_embed", "cls_token", "pos_embed", "mask_token", *frozen_block_names],
        "trainable_module_names": [*trainable_block_names, "norm"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_pct": 100.0 * trainable_params / total_params,
    }


def print_freeze_report(report: dict[str, object]) -> None:
    print(
        f"[3DINO] total_params={report['total_params']:,} "
        f"trainable_params={report['trainable_params']:,} "
        f"frozen_params={report['frozen_params']:,} "
        f"({report['trainable_pct']:.2f}% trainable)",
        flush=True,
    )
    print(f"[3DINO] frozen blocks: {report['frozen_blocks']}", flush=True)
    print(f"[3DINO] trainable blocks: {report['trainable_blocks']}", flush=True)
    print(f"[3DINO] frozen modules: {report['frozen_module_names']}", flush=True)
    print(f"[3DINO] trainable modules: {report['trainable_module_names']}", flush=True)


def percentile_normalize(x: torch.Tensor) -> torch.Tensor:
    """Official preprocessing (verified from ``notebooks/basic_model_use.ipynb``):
    rescale the [0.05, 99.95] percentile range to [-1, 1], with hard clipping.
    Matches the SSL pretraining convention of normalizing each crop/window
    independently: for a batch ``(N, ...)`` the percentiles are computed
    per-sample (per window), not pooled across the batch."""
    x = x.float()
    flat = x.reshape(x.shape[0], -1)
    min_val = torch.quantile(flat, 0.0005, dim=1).view(-1, *([1] * (x.dim() - 1)))
    max_val = torch.quantile(flat, 0.9995, dim=1).view(-1, *([1] * (x.dim() - 1)))
    x = (x - min_val) / (max_val - min_val).clamp_min(1e-6)
    return torch.clip(x * 2 - 1, -1, 1)


def sliding_window_starts_3d(shape: tuple[int, int, int], window: int, stride: int) -> list[tuple[int, int, int]]:
    """3D sliding-window start coordinates covering the full volume, including a
    flush final window per axis when the dimension isn't stride-divisible
    (same flush-last-window convention as the RadJEPA baseline)."""

    def axis_starts(size: int) -> list[int]:
        if size <= window:
            return [0]
        starts = list(range(0, size - window + 1, stride))
        last_start = size - window
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    axis_ranges = [axis_starts(size) for size in shape]
    return list(itertools.product(*axis_ranges))


def extract_windows(volume: torch.Tensor, window: int, stride: int) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
    """volume: (D, H, W) -> (n_windows, 1, window, window, window), zero-padded
    when a native dimension is smaller than the window size."""
    d, h, w = volume.shape
    starts = sliding_window_starts_3d((d, h, w), window, stride)
    windows = []
    for zs, ys, xs in starts:
        crop = volume[zs : zs + window, ys : ys + window, xs : xs + window]
        pad = [0, window - crop.shape[2], 0, window - crop.shape[1], 0, window - crop.shape[0]]
        if any(p > 0 for p in pad):
            crop = torch.nn.functional.pad(crop, pad, mode="constant", value=0.0)
        windows.append(crop)
    stacked = torch.stack(windows, dim=0).unsqueeze(1)
    return stacked, starts
