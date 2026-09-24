"""Extract frozen COLIPRI-CRM embeddings for every LUNA16 patient.

Runs the official ``colipri`` package (official checkpoint + official
``Processor``) over the *raw* LUNA16 CT volumes (native spacing, raw HU),
letting COLIPRI perform its own reorientation / resampling to 2mm isotropic /
HU clamping / rescaling / center-crop-or-pad to 192x192x192, exactly as
documented on https://huggingface.co/microsoft/colipri.

For each patient (seriesuid) this saves two 768-D representations:
  - ``pooled``: the official pooled+projected embedding
    (``model.encode_image(batch, pool=True, project=True)``).
  - ``dense_maxpool``: the dense projected feature grid
    (``model.encode_image(batch, project=True, pool=False)``, shape
    ``(768, 24, 24, 24)``) reduced with a channel-wise global max over the
    spatial grid.

Since COLIPRI is frozen and patient identity/labels are shared across all 10
Adaptive-RBF folds (verified: every ``luna16_classification_fold{0-9}.csv``
contains the same 888 seriesuids/labels, only ``split`` differs), embeddings
are extracted once for the full patient set and cached to disk; the linear
probes then reuse this cache across all folds and both representations.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def find_raw_ct_path(raw_root: Path, image_path: str) -> Path:
    """Map a processed-volume ``image_path`` (e.g. ``subset0/<uid>/<uid>_volume.nii.gz``)
    to the corresponding raw LUNA16 ``.mhd`` file (native spacing, raw HU)."""
    subset = image_path.split("/")[0]
    seriesuid = Path(image_path).name.removesuffix("_volume.nii.gz")
    return raw_root / subset / subset / f"{seriesuid}.mhd"


def load_patient_table(fold_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(fold_csv)
    required = {"seriesuid", "image_path", "target", "target_name"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{fold_csv} is missing expected columns: {missing}")
    df = df.drop_duplicates(subset="seriesuid").reset_index(drop=True)
    return df


@torch.no_grad()
def encode_batch(model, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Run the frozen encoder's official pooled path and dense-projected path.

    Two separate ``encode_image`` calls are used (matching the official API
    rather than hand-decomposing the model internals): one for the official
    pool=True/project=True embedding, one for the dense project=True/pool=False
    grid that we then reduce ourselves with a channel-wise global max pool.
    """
    pooled = model.encode_image(batch, pool=True, project=True)
    dense = model.encode_image(batch, project=True, pool=False)
    dense_maxpool = torch.amax(dense, dim=(2, 3, 4))
    return pooled.float().cpu().numpy(), dense_maxpool.float().cpu().numpy()


# --- CPU-bound preprocessing, parallelized across worker processes --------

_WORKER_PROCESSOR = None


def _init_worker(threads_per_worker: int) -> None:
    global _WORKER_PROCESSOR
    import os

    # Each preprocessing task (SimpleITK resample, etc.) otherwise tries to use
    # all available cores itself; with many worker processes that oversubscribes
    # the CPU and makes the pool *slower* than a single process. Cap per-worker
    # threading so the parallelism comes from the process count instead.
    for var in ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads_per_worker)

    import torch as _torch

    _torch.set_num_threads(threads_per_worker)

    from colipri import get_processor

    _WORKER_PROCESSOR = get_processor(image_only=True)


def _preprocess_one(task: tuple[str, str]) -> tuple[str, np.ndarray | None, str | None]:
    seriesuid, ct_path = task
    try:
        images = _WORKER_PROCESSOR.process_images(ct_path)
        batch = _WORKER_PROCESSOR.to_images_batch(images)
        return seriesuid, batch.numpy(), None
    except Exception as exc:  # noqa: BLE001 - reported back to the main process
        return seriesuid, None, str(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen COLIPRI-CRM embeddings for LUNA16 patients.")
    parser.add_argument(
        "--fold-csv",
        default="data/processed/cv_splits/luna16_classification_fold0.csv",
        help="Any single fold CSV; all 10 folds share the same 888 patients/labels (only split differs).",
    )
    parser.add_argument("--raw-root", default="data/raw/LUNA16/subsets")
    parser.add_argument("--output-dir", default="outputs/luna16_colipri_3d/embeddings")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if a cached .npz already exists.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N patients (debugging).")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Worker processes for CPU-bound COLIPRI preprocessing (resample/clamp/rescale/crop-pad).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    per_patient_dir = output_dir / "per_patient"
    per_patient_dir.mkdir(parents=True, exist_ok=True)

    df = load_patient_table(Path(args.fold_csv))
    if args.limit is not None:
        df = df.head(args.limit)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    from colipri import get_model, get_processor  # deferred: heavy import + HF download

    model = get_model(image_only=True).to(device).eval()
    processor = get_processor(image_only=True)

    config = {
        "checkpoint": "microsoft/colipri (COLIPRI-CRM, image_only=True)",
        "colipri_package_version": __import__("colipri").__dict__.get("__version__", "unknown"),
        "processor_repr": repr(processor),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hostname": platform.node(),
        "fold_csv_used_for_patient_list": str(args.fold_csv),
        "raw_root": str(args.raw_root),
        "n_patients": int(len(df)),
        "representations": {
            "pooled": "model.encode_image(batch, pool=True, project=True) -> (768,)",
            "dense_maxpool": "amax(model.encode_image(batch, project=True, pool=False), dim=spatial) -> (768,), grid=24x24x24",
        },
    }
    with (output_dir / "extraction_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    raw_root = Path(args.raw_root)
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    row_by_uid = {str(row["seriesuid"]): row for _, row in df.iterrows()}

    def register(seriesuid: str, pooled: np.ndarray, dense_maxpool: np.ndarray) -> None:
        row = row_by_uid[seriesuid]
        records.append(
            {
                "seriesuid": seriesuid,
                "lidc_id": row.get("lidc_id"),
                "subset": str(row["image_path"]).split("/")[0],
                "target": int(row["target"]),
                "target_name": row["target_name"],
                "image_path": row["image_path"],
            }
        )

    start = time.time()
    n_done = 0
    total = len(df)

    tasks: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        seriesuid = str(row["seriesuid"])
        cache_path = per_patient_dir / f"{seriesuid}.npz"
        if cache_path.exists() and not args.overwrite:
            cached = np.load(cache_path)
            register(seriesuid, cached["pooled"], cached["dense_maxpool"])
            n_done += 1
            continue
        ct_path = find_raw_ct_path(raw_root, str(row["image_path"]))
        if not ct_path.exists():
            failures.append({"seriesuid": seriesuid, "reason": f"raw CT not found: {ct_path}"})
            print(f"MISSING {seriesuid}: {ct_path}", flush=True)
            continue
        tasks.append((seriesuid, str(ct_path)))

    print(f"{n_done}/{total} patients already cached; preprocessing {len(tasks)} remaining with "
          f"{args.num_workers} worker process(es)...", flush=True)

    if tasks:
        num_workers = max(1, args.num_workers)
        threads_per_worker = max(1, multiprocessing.cpu_count() // num_workers)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(threads_per_worker,)) as pool:
            for seriesuid, batch_np, error in pool.imap_unordered(_preprocess_one, tasks, chunksize=1):
                if error is not None:
                    failures.append({"seriesuid": seriesuid, "reason": error})
                    print(f"FAILED (preprocess) {seriesuid}: {error}", flush=True)
                    continue

                batch = torch.from_numpy(batch_np).to(device)
                pooled, dense_maxpool = encode_batch(model, batch)
                pooled, dense_maxpool = pooled[0], dense_maxpool[0]

                cache_path = per_patient_dir / f"{seriesuid}.npz"
                tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp.npz")
                np.savez(tmp_path, pooled=pooled, dense_maxpool=dense_maxpool)
                tmp_path.replace(cache_path)

                register(seriesuid, pooled, dense_maxpool)
                n_done += 1
                if n_done % 25 == 0 or n_done == total:
                    elapsed = time.time() - start
                    print(f"[{n_done}/{total}] processed, elapsed={elapsed:.1f}s", flush=True)

    if not records:
        raise RuntimeError("No embeddings were extracted; check --raw-root and the fold CSV path.")

    first_cached = np.load(per_patient_dir / f"{records[0]['seriesuid']}.npz")
    n_pooled = first_cached["pooled"].shape[0]
    n_dense = first_cached["dense_maxpool"].shape[0]

    index_df = pd.DataFrame(records)
    index_df.to_csv(output_dir / "colipri_embeddings_index.csv", index=False)

    pooled_matrix = np.zeros((len(records), n_pooled), dtype=np.float32)
    dense_matrix = np.zeros((len(records), n_dense), dtype=np.float32)
    for i, rec in enumerate(records):
        cached = np.load(per_patient_dir / f"{rec['seriesuid']}.npz")
        pooled_matrix[i] = cached["pooled"]
        dense_matrix[i] = cached["dense_maxpool"]

    np.savez(
        output_dir / "colipri_embeddings.npz",
        seriesuid=index_df["seriesuid"].to_numpy(),
        target=index_df["target"].to_numpy(),
        pooled=pooled_matrix,
        dense_maxpool=dense_matrix,
    )

    if failures:
        with (output_dir / "extraction_failures.json").open("w") as handle:
            json.dump(failures, handle, indent=2)

    print(
        f"Done. {len(records)} patients embedded, {len(failures)} failed. "
        f"Consolidated file: {output_dir / 'colipri_embeddings.npz'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
