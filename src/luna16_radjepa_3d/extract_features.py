"""Extract frozen Rad-JEPA-3D embeddings for every LUNA16 patient.

Rad-JEPA-3D (H-Mamba hybrid encoder, checkpoint from the sibling project
``lung-jepa-world-model``) expects volumes shaped ``(1, 32, 256, 256)``,
float32, HU-windowed and rescaled to ``[0, 1]`` (the exact preprocessing
convention already established by that project's
``scripts/prepare_radjepa_volumes.py`` / ``prepare_radjepa_sliding_volume.py``).

This script must be run with the sibling project's Python interpreter
(``/home/domenico/lung-jepa-world-model/.venv/bin/python3``), which has the
``mamba_ssm``/``causal_conv1d`` CUDA extensions the encoder depends on; this
repo's own ``myenv`` intentionally keeps a newer, incompatible ``torch`` and
is never modified for this baseline (see ``README.md``).

For each patient (seriesuid) this computes, from the frozen encoder's
mean-pooled token embedding (``encoder(x, indices=None).mean(dim=1)`` ->
384-D, exactly the convention used by
``lung-jepa-world-model/scripts/extract_radjepa_embeddings.py``):

  - ``resize32``: one embedding from the whole scan resized (zoom, order=1)
    to (32, 256, 256) -- a single "window" covering the full volume.
  - a variable-length set of ``sliding`` window embeddings: the scan resized
    in-plane only to (native_depth, 256, 256), then split into 32-slice
    windows with stride 16 (plus a final window flush with the last slice so
    the tail is never dropped when depth isn't stride-aligned).

Patient identity/labels are shared across all 10 Adaptive-RBF folds (same
invariant already verified for the COLIPRI baseline), so embeddings are
extracted once for the full patient set and cached to disk; the patient-level
pooling strategies (resize32 / sliding-mean / sliding-max / sliding-mean+max)
are all derived later, from this cache, without re-running the encoder.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.luna16_colipri_3d.extract_features import find_raw_ct_path, load_patient_table  # noqa: E402

DEFAULT_RADJEPA_CODE_DIR = Path("/home/domenico/lung-jepa-world-model/external/RadJepa")
DEFAULT_HF_MODEL_DIR = Path("/home/domenico/lung-jepa-world-model/external/Rad-Jepa-3D-hf")
WINDOW_DEPTH = 32
WINDOW_SIZE_HW = 256


def hu_window_normalize(volume: np.ndarray, window_min: float, window_max: float) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    volume = np.nan_to_num(volume, nan=window_min, posinf=window_max, neginf=window_min)
    volume = np.clip(volume, window_min, window_max)
    return (volume - window_min) / max(window_max - window_min, 1e-6)


def resize_dhw(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    from scipy.ndimage import zoom

    factors = [t / c for t, c in zip(target_shape, volume.shape)]
    return zoom(volume, zoom=factors, order=1).astype(np.float32, copy=False)


def resize_hw(volume: np.ndarray, size: int) -> np.ndarray:
    from scipy.ndimage import zoom

    factors = (1.0, size / volume.shape[1], size / volume.shape[2])
    return zoom(volume, zoom=factors, order=1).astype(np.float32, copy=False)


def sliding_window_starts(depth: int, window: int, stride: int) -> list[int]:
    if depth <= window:
        return [0]
    starts = list(range(0, depth - window + 1, stride))
    last_start = depth - window
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_encoder(radjepa_code_dir: Path, config_path: Path, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(radjepa_code_dir.resolve()))
    from vjepa_config import VJEPAConfig
    from vjepa_model import MambaEncoder

    with config_path.open() as handle:
        hf_cfg = json.load(handle)

    cfg = VJEPAConfig()
    cfg.dim = int(hf_cfg["hidden_size"])
    cfg.n_layers = int(hf_cfg["num_hidden_layers"])
    cfg.patch_size = tuple(hf_cfg["patch_size"])
    cfg.volume_size = tuple(hf_cfg["input_shape"][1:])
    cfg.use_hybrid = True
    cfg.router_mode = hf_cfg.get("router_mode", "layer")
    cfg.use_morton_scan = hf_cfg.get("scan_order", "raster") == "morton"
    cfg.use_rope = hf_cfg.get("position_encoding", "rope_3d") == "rope_3d"

    encoder = MambaEncoder(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("target_encoder") or checkpoint.get("context_encoder") or checkpoint
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing_keys={len(missing)} unexpected_keys={len(unexpected)}", flush=True)

    encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, cfg, hf_cfg


@torch.no_grad()
def embed_window(encoder, window: np.ndarray, device: torch.device) -> np.ndarray:
    """window: (32, 256, 256) float32 in [0, 1] -> mean-pooled (dim,) embedding."""
    tensor = torch.from_numpy(window).unsqueeze(0).unsqueeze(0).to(device)
    tokens = encoder(tensor, indices=None)
    pooled = tokens.mean(dim=1)
    return pooled.squeeze(0).float().cpu().numpy()


def load_and_prepare(ct_path: Path, window_min: float, window_max: float) -> tuple[np.ndarray, np.ndarray, int]:
    image = sitk.ReadImage(str(ct_path))
    volume = sitk.GetArrayFromImage(image)  # (D, H, W), raw HU
    volume = hu_window_normalize(volume, window_min, window_max)

    native_depth = volume.shape[0]
    resize32_volume = resize_dhw(volume, (WINDOW_DEPTH, WINDOW_SIZE_HW, WINDOW_SIZE_HW))
    sliding_volume = resize_hw(volume, WINDOW_SIZE_HW)  # (native_depth, 256, 256)
    return resize32_volume, sliding_volume, native_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen Rad-JEPA-3D embeddings for LUNA16 patients.")
    parser.add_argument(
        "--fold-csv",
        default="data/processed/cv_splits/luna16_classification_fold0.csv",
        help="Any single fold CSV; all 10 folds share the same 888 patients/labels (only split differs).",
    )
    parser.add_argument("--raw-root", default="data/raw/LUNA16/subsets")
    parser.add_argument("--output-dir", default="outputs/luna16_radjepa_3d/embeddings")
    parser.add_argument("--radjepa-code-dir", type=Path, default=DEFAULT_RADJEPA_CODE_DIR)
    parser.add_argument("--hf-model-dir", type=Path, default=DEFAULT_HF_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--window-min", type=float, default=-1000.0)
    parser.add_argument("--window-max", type=float, default=400.0)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N patients (debugging).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    per_patient_dir = output_dir / "per_patient"
    per_patient_dir.mkdir(parents=True, exist_ok=True)

    df = load_patient_table(Path(args.fold_csv))
    if args.limit is not None:
        df = df.head(args.limit)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint_path = args.checkpoint or args.hf_model_dir / "encoder" / "encoder.pt"
    config_path = args.config or args.hf_model_dir / "config.json"
    encoder, cfg, hf_cfg = build_encoder(args.radjepa_code_dir, config_path, checkpoint_path, device)

    config = {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "radjepa_code_dir": str(args.radjepa_code_dir),
        "hf_config": hf_cfg,
        "embedding_dim": int(cfg.dim),
        "window_shape": [WINDOW_DEPTH, WINDOW_SIZE_HW, WINDOW_SIZE_HW],
        "window_stride": args.window_stride,
        "hu_window": [args.window_min, args.window_max],
        "scan_embedding_definition": "encoder(x, indices=None).mean(dim=1) -> (embedding_dim,), "
        "same convention as lung-jepa-world-model/scripts/extract_radjepa_embeddings.py",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hostname": platform.node(),
        "fold_csv_used_for_patient_list": str(args.fold_csv),
        "raw_root": str(args.raw_root),
        "n_patients": int(len(df)),
        "seed": args.seed,
    }
    with (output_dir / "extraction_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    raw_root = Path(args.raw_root)
    records: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    start_time = time.time()
    for i, row in df.iterrows():
        seriesuid = str(row["seriesuid"])
        cache_path = per_patient_dir / f"{seriesuid}.npz"

        if cache_path.exists() and not args.overwrite:
            cached = np.load(cache_path)
            n_windows = int(cached["window_starts"].shape[0])
        else:
            ct_path = find_raw_ct_path(raw_root, str(row["image_path"]))
            if not ct_path.exists():
                failures.append({"seriesuid": seriesuid, "reason": f"raw CT not found: {ct_path}"})
                print(f"[{i + 1}/{len(df)}] MISSING {seriesuid}: {ct_path}", flush=True)
                continue
            try:
                resize32_volume, sliding_volume, native_depth = load_and_prepare(
                    ct_path, args.window_min, args.window_max
                )
                resize32_embedding = embed_window(encoder, resize32_volume, device)

                starts = sliding_window_starts(native_depth, WINDOW_DEPTH, args.window_stride)
                window_embeddings = []
                for w_start in starts:
                    w_end = min(w_start + WINDOW_DEPTH, native_depth)
                    window = sliding_volume[w_start : w_start + WINDOW_DEPTH]
                    if window.shape[0] < WINDOW_DEPTH:
                        pad = np.zeros((WINDOW_DEPTH - window.shape[0], WINDOW_SIZE_HW, WINDOW_SIZE_HW), dtype=np.float32)
                        window = np.concatenate([window, pad], axis=0)
                    window_embeddings.append(embed_window(encoder, window, device))
                window_embeddings = np.stack(window_embeddings, axis=0)
                window_starts = np.array(starts, dtype=np.int32)
                window_ends = np.array(
                    [min(s + WINDOW_DEPTH, native_depth) for s in starts], dtype=np.int32
                )
            except Exception as exc:  # noqa: BLE001 - keep extracting remaining patients
                failures.append({"seriesuid": seriesuid, "reason": str(exc)})
                print(f"[{i + 1}/{len(df)}] FAILED {seriesuid}: {exc}", flush=True)
                continue

            tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp.npz")
            np.savez(
                tmp_path,
                resize32_embedding=resize32_embedding,
                window_embeddings=window_embeddings,
                window_starts=window_starts,
                window_ends=window_ends,
                native_depth=np.array(native_depth, dtype=np.int32),
            )
            tmp_path.replace(cache_path)
            n_windows = int(window_starts.shape[0])

        records.append(
            {
                "seriesuid": seriesuid,
                "lidc_id": row.get("lidc_id"),
                "subset": str(row["image_path"]).split("/")[0],
                "target": int(row["target"]),
                "target_name": row["target_name"],
                "image_path": row["image_path"],
                "num_windows": n_windows,
            }
        )
        cached = np.load(cache_path)
        for w_idx in range(n_windows):
            window_rows.append(
                {
                    "seriesuid": seriesuid,
                    "window_index": w_idx,
                    "window_start": int(cached["window_starts"][w_idx]),
                    "window_end": int(cached["window_ends"][w_idx]),
                    "num_windows": n_windows,
                    "embedding_path": str(cache_path),
                }
            )

        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            elapsed = time.time() - start_time
            print(f"[{i + 1}/{len(df)}] processed, elapsed={elapsed:.1f}s", flush=True)

    if not records:
        raise RuntimeError("No embeddings were extracted; check --raw-root and the fold CSV path.")

    index_df = pd.DataFrame(records)
    index_df.to_csv(output_dir / "radjepa_embeddings_index.csv", index=False)
    pd.DataFrame(window_rows).to_csv(output_dir / "radjepa_window_manifest.csv", index=False)

    if failures:
        with (output_dir / "extraction_failures.json").open("w") as handle:
            json.dump(failures, handle, indent=2)

    print(
        f"Done. {len(records)} patients embedded, {len(failures)} failed. "
        f"Index: {output_dir / 'radjepa_embeddings_index.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
