from __future__ import annotations

import argparse
import os
import sys
import traceback
import types
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _install_saliency_import_shims() -> None:
    """Let this non-Hydra entrypoint import saliency.py helper functions."""
    hydra_stub = types.ModuleType("hydra")

    def main(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    hydra_stub.main = main
    sys.modules["hydra"] = hydra_stub

    if "omegaconf" not in sys.modules:
        omegaconf_stub = types.ModuleType("omegaconf")

        class OmegaConf:
            @staticmethod
            def set_struct(*_args, **_kwargs):
                return None

        omegaconf_stub.DictConfig = dict
        omegaconf_stub.OmegaConf = OmegaConf
        sys.modules["omegaconf"] = omegaconf_stub


_install_saliency_import_shims()

from src.luna16_synthetic_2d.saliency import (
    LUNA16NiftiDataset,
    compute_lung_coverage,
    fit_surface_grid,
    get_lung_mask,
    mid_slice_plane,
)


PREDICTION_COLUMNS = ("seriesuid", "coordZ", "coordY", "coordX", "radius", "probability")


def find_fold_dir(pred_root: Path, fold: int) -> Path | None:
    candidates = [
        pred_root / f"fold_{fold}",
        pred_root / f"fold{fold}",
        pred_root / f"scpmnet_paper_luna16_fold{fold}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(pred_root.glob(f"*fold{fold}"))
    return matches[0] if matches else None


def prediction_csv_for_fold(pred_root: Path, fold: int, prediction_name: str) -> Path:
    fold_dir = find_fold_dir(pred_root, fold)
    if fold_dir is None:
        raise FileNotFoundError(f"Could not find detector output directory for fold {fold} under {pred_root}")

    candidates = [
        fold_dir / "predictions" / prediction_name,
        fold_dir / prediction_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {prediction_name} for fold {fold} in {fold_dir}")


def load_top_detections(prediction_csv: Path, top_k: int, min_probability: float | None) -> dict[str, pd.DataFrame]:
    predictions = pd.read_csv(prediction_csv)
    missing = [col for col in PREDICTION_COLUMNS if col not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction CSV {prediction_csv} is missing columns: {missing}")

    predictions = predictions.dropna(subset=["seriesuid", "coordZ", "coordY", "coordX", "probability"]).copy()
    predictions["seriesuid"] = predictions["seriesuid"].astype(str)
    predictions["probability"] = predictions["probability"].astype(float)
    if min_probability is not None:
        predictions = predictions[predictions["probability"] >= float(min_probability)]

    by_series: dict[str, pd.DataFrame] = {}
    for seriesuid, rows in predictions.groupby("seriesuid", sort=False):
        by_series[str(seriesuid)] = (
            rows.sort_values("probability", ascending=False)
            .head(int(top_k))
            .reset_index(drop=True)
        )
    return by_series


def detection_control_points(
    detections: pd.DataFrame,
    num_contour_points: int,
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    depth, height, width = volume_shape
    points: list[list[float]] = []
    num_contour_points = max(0, int(num_contour_points))

    for _, detection in detections.iterrows():
        z = float(detection["coordZ"])
        y = float(detection["coordY"])
        x = float(detection["coordX"])
        radius = float(detection["radius"]) if pd.notna(detection.get("radius")) else 0.0
        radius = max(radius, 1.0)

        nodule_points = [[z, y, x]]
        if num_contour_points > 0:
            angles = np.linspace(0.0, 2.0 * np.pi, num_contour_points, endpoint=False)
            for angle in angles:
                nodule_points.append(
                    [
                        z,
                        y + radius * float(np.sin(angle)),
                        x + radius * float(np.cos(angle)),
                    ]
                )

        clipped = np.asarray(nodule_points, dtype=np.float32)
        clipped[:, 0] = np.clip(clipped[:, 0], 0, depth - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
        clipped[:, 2] = np.clip(clipped[:, 2], 0, width - 1)
        points.extend(clipped.tolist())

    return np.asarray(points, dtype=np.float32)


def normalize_surface_image(output_image: np.ndarray, lung_window_center: float, lung_window_width: float) -> np.ndarray:
    if output_image.min() >= 0 and output_image.max() <= 255:
        output_image = output_image.astype(np.float32) / 255.0
    else:
        window_min = lung_window_center - (lung_window_width / 2)
        window_max = lung_window_center + (lung_window_width / 2)
        output_image = np.clip(output_image, window_min, window_max)
        output_image = (output_image - window_min) / (window_max - window_min)
    return np.clip(output_image, 0, 1)


def unpack_sample(sample):
    if len(sample) == 4:
        img, label, patient_id, saved_lung_mask = sample
    elif len(sample) == 3:
        img, label, patient_id = sample
        saved_lung_mask = None
    else:
        raise ValueError(f"Expected dataset sample length 3 or 4, got {len(sample)}")
    return img, label, str(patient_id), saved_lung_mask


def save_detector_surfaces(args: argparse.Namespace, dataset: LUNA16NiftiDataset, detections_by_series: dict[str, pd.DataFrame]) -> None:
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for idx in range(len(dataset)):
        patient_id = "<unknown>"
        try:
            img, _label, patient_id, saved_lung_mask = unpack_sample(dataset[idx])
            detections = detections_by_series.get(patient_id)
            if detections is None or detections.empty:
                message = f"No detector predictions found for {patient_id}."
                if args.fallback_mid_slice:
                    print(f"Warning: {message} Using mid-slice fallback.")
                else:
                    print(f"Warning: {message} Skipping.")
                    continue

            print(f"Processing patient: {patient_id}")
            h, w = img.shape[2:]
            depth = int(img.shape[0])
            volume_np = img.numpy()[:, 0, :, :]

            lung_mask = None
            if args.use_saved_lung_masks and saved_lung_mask is not None:
                saved_lung_mask_np = saved_lung_mask.numpy() > 0
                if np.any(saved_lung_mask_np):
                    lung_mask = saved_lung_mask_np
                    print(f"Using saved lung mask for {patient_id}: {int(lung_mask.sum())} voxels.")
                else:
                    print(f"Warning: saved lung mask is empty for {patient_id}; falling back to runtime lung mask.")

            if lung_mask is None and (args.use_lung_volume_anchors or args.report_lung_coverage):
                lung_mask = get_lung_mask(
                    volume_np,
                    model_name=args.lungmask_model_name,
                    force_cpu=args.lungmask_force_cpu,
                    method=args.lung_mask_method,
                    normalized_air_threshold=args.normalized_air_threshold,
                    hu_air_min=args.hu_air_min,
                    hu_air_max=args.hu_air_max,
                    body_threshold_percentile=args.body_threshold_percentile,
                    lung_component_count=args.lung_component_count,
                )

            if detections is None or detections.empty:
                matrix = mid_slice_plane(depth, h, w)
                selected = pd.DataFrame(columns=PREDICTION_COLUMNS)
            else:
                selected = detections.head(args.top_k).reset_index(drop=True)
                matrix = detection_control_points(
                    selected,
                    num_contour_points=args.num_contour_points,
                    volume_shape=(depth, h, w),
                )
                print(
                    f"Using {len(matrix)} detector control points "
                    f"({1 + max(0, int(args.num_contour_points))} per detection) from fold {args.fold}; "
                    f"top probability={float(selected['probability'].max()):.4f}."
                )

            matrix, point_labels, z_surface_float, z_surface = fit_surface_grid(
                matrix,
                h,
                w,
                depth,
                args.num_boundary_anchors,
                args.rbf_smooth,
                lung_mask=lung_mask,
                patient_id=patient_id,
                use_lung_volume_anchors=args.use_lung_volume_anchors,
                lung_anchor_erode_iterations=args.lung_anchor_erode_iterations,
                snap_surface_to_lung=args.snap_surface_to_lung,
                anchor_min_lung_area_fraction=args.anchor_min_lung_area_fraction,
            )

            lung_coverage = ""
            if args.report_lung_coverage and lung_mask is not None and np.any(lung_mask):
                lung_coverage = float(compute_lung_coverage(z_surface, lung_mask))
                print(f"Lung coverage: {lung_coverage:.3f}")

            current_save_dir = save_path / patient_id
            current_save_dir.mkdir(parents=True, exist_ok=True)

            surface_png = current_save_dir / f"surface_{patient_id}.png"
            rows, cols = np.indices((h, w))
            img_np = img.numpy()
            output_image = img_np[z_surface, 0, rows, cols]
            output_image = normalize_surface_image(
                output_image,
                lung_window_center=args.lung_window_center,
                lung_window_width=args.lung_window_width,
            )
            Image.fromarray((output_image * 255).astype(np.uint8)).convert("RGB").save(surface_png)

            if args.save_surface_grid:
                np.save(current_save_dir / f"surface_grid_float_{patient_id}.npy", z_surface_float.astype(np.float32))
                np.save(current_save_dir / f"surface_grid_int_{patient_id}.npy", z_surface.astype(np.int16))
                np.save(current_save_dir / f"control_points_{patient_id}.npy", matrix.astype(np.float32))
                np.save(current_save_dir / f"point_labels_{patient_id}.npy", np.asarray(point_labels))
                if not selected.empty:
                    selected.to_csv(current_save_dir / f"detector_top{args.top_k}_{patient_id}.csv", index=False)

            manifest_rows.append(
                {
                    "fold": int(args.fold),
                    "split": args.split,
                    "seriesuid": patient_id,
                    "surface_png": str(surface_png),
                    "num_detections": int(len(selected)),
                    "num_control_points": int(len(matrix)),
                    "control_points_per_detection": int(1 + max(0, int(args.num_contour_points))),
                    "requested_top_k": int(args.top_k),
                    "top_probability": float(selected["probability"].max()) if not selected.empty else "",
                    "mean_detection_z": float(selected["coordZ"].mean()) if not selected.empty else "",
                    "min_detection_z": float(selected["coordZ"].min()) if not selected.empty else "",
                    "max_detection_z": float(selected["coordZ"].max()) if not selected.empty else "",
                    "lung_coverage": lung_coverage,
                }
            )
        except Exception:
            print(f"\nERROR ITEM {idx}")
            print(f"PATIENT: {patient_id}")
            traceback.print_exc()

    if manifest_rows:
        manifest_path = save_path / f"manifest_fold{args.fold}_{args.split}.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        print(f"Wrote manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LUNA16 saliency-style synthetic 2D surfaces from the top-k "
            "SCPMNet detections for the matching CV fold detector."
        )
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--split", default="test", help="Use test to pair each LUNA16 subset with its matching fold detector.")
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--processed-dir", default="data/LUNA16_preprocessed")
    parser.add_argument("--pred-root", default="outputs/scpmnet_luna16_10fold")
    parser.add_argument("--prediction-name", default="test_predictions.csv")
    parser.add_argument("--save-path", default="outputs/luna16_saliency_synthetic_detector_top5")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-contour-points", type=int, default=4)
    parser.add_argument("--min-probability", type=float, default=None)
    parser.add_argument("--fallback-mid-slice", action="store_true")
    parser.add_argument("--save-surface-grid", action="store_true")
    parser.add_argument("--report-lung-coverage", action="store_true")

    parser.add_argument("--rbf-smooth", type=float, default=0.1)
    parser.add_argument("--num-boundary-anchors", type=int, default=24)
    parser.add_argument("--use-lung-volume-anchors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lung-anchor-erode-iterations", type=int, default=1)
    parser.add_argument("--snap-surface-to-lung", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anchor-min-lung-area-fraction", type=float, default=0.35)
    parser.add_argument("--use-saved-lung-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lung-mask-method", default="body_threshold")
    parser.add_argument("--normalized-air-threshold", type=float, default=0.35)
    parser.add_argument("--hu-air-min", type=float, default=-1000.0)
    parser.add_argument("--hu-air-max", type=float, default=-320.0)
    parser.add_argument("--body-threshold-percentile", type=float, default=1.0)
    parser.add_argument("--lung-component-count", type=int, default=2)
    parser.add_argument("--lungmask-model-name", default="R231")
    parser.add_argument("--lungmask-force-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lung-window-center", type=float, default=-600.0)
    parser.add_argument("--lung-window-width", type=float, default=1500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    prediction_csv = prediction_csv_for_fold(Path(args.pred_root), args.fold, args.prediction_name)
    print(f"Fold {args.fold} detector predictions: {prediction_csv}")
    print(f"Fold {args.fold} CSV: {args.csv_file}")
    print(f"Processing split: {args.split}")

    detections_by_series = load_top_detections(prediction_csv, args.top_k, args.min_probability)
    dataset = LUNA16NiftiDataset(
        args.csv_file,
        args.processed_dir,
        split=args.split,
        return_mask=False,
        return_lung_mask=args.use_saved_lung_masks,
        only_no_nodules=False,
    )
    save_detector_surfaces(args, dataset, detections_by_series)


if __name__ == "__main__":
    main()
