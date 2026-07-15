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
    load_empirical_nodule_distribution,
    mid_slice_plane,
    sample_empirical_pseudo_nodules,
    sample_pseudo_regions_inside_lung,
)


PREDICTION_COLUMNS = ("seriesuid", "coordZ", "coordY", "coordX", "radius", "probability")
MALIGNANCY_COLUMNS = (
    "malignancy",
    "mean_malignancy",
    "nodule_malignancy",
    "nodule_mean_malignancy",
    "nodule_annotation_mean_malignancy",
)


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
            .pipe(remove_detections_overlapping)
            .head(int(top_k))
            .reset_index(drop=True)
        )
    return by_series


def detection_priority_column(detections: pd.DataFrame) -> str:
    for column in MALIGNANCY_COLUMNS:
        if column in detections.columns:
            return column
    return "probability"


def remove_detections_overlapping(detections: pd.DataFrame) -> pd.DataFrame:
    """Keep the most malignant detection for each rounded y/x location when available."""
    if detections.empty:
        return detections

    priority_column = detection_priority_column(detections)
    ordered = detections.copy()
    ordered[priority_column] = pd.to_numeric(ordered[priority_column], errors="coerce")
    ordered["probability"] = pd.to_numeric(ordered["probability"], errors="coerce")
    sort_columns = [priority_column] if priority_column == "probability" else [priority_column, "probability"]
    ordered = ordered.sort_values(sort_columns, ascending=False).copy()
    yx = np.round(ordered[["coordY", "coordX"]].to_numpy(dtype=float)).astype(int)
    keep_mask = []
    kept_yx = set()
    removed = 0

    for y, x in yx:
        key = (int(y), int(x))
        if key in kept_yx:
            keep_mask.append(False)
            removed += 1
            continue
        kept_yx.add(key)
        keep_mask.append(True)

    if removed > 0:
        seriesuid = str(ordered["seriesuid"].iloc[0]) if "seriesuid" in ordered.columns else "<unknown>"
        print(
            f"  Removed {removed} overlapping detector candidates for {seriesuid} "
            f"with duplicate y/x using {priority_column} priority."
        )

    return ordered.loc[keep_mask].reset_index(drop=True)


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


def detection_shepard_disks(detections: pd.DataFrame, volume_shape: tuple[int, int, int]) -> np.ndarray:
    depth, height, width = volume_shape
    disks: list[list[float]] = []
    for _, detection in detections.iterrows():
        z = float(np.clip(float(detection["coordZ"]), 0, depth - 1))
        y = float(np.clip(float(detection["coordY"]), 0, height - 1))
        x = float(np.clip(float(detection["coordX"]), 0, width - 1))
        radius = float(detection["radius"]) if pd.notna(detection.get("radius")) else 0.0
        disks.append([x, y, z, max(radius, 1.0)])
    return np.asarray(disks, dtype=np.float32)


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


def detector_negative_surface(
    args: argparse.Namespace,
    patient_id: str,
    volume_np: np.ndarray,
    lung_mask: np.ndarray | None,
    h: int,
    w: int,
    depth: int,
    empirical_distribution: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    pseudo_max_attempts = max(1, int(args.pseudo_max_attempts))
    best_candidate = None
    best_lung_coverage = -1.0

    for attempt in range(pseudo_max_attempts):
        print(f"Detector-negative pseudo-region attempt: {attempt + 1}/{pseudo_max_attempts}")
        if empirical_distribution is not None:
            candidate_matrix, candidate_shepard_detections = sample_empirical_pseudo_nodules(
                patient_id,
                volume_np.shape,
                empirical_distribution,
                lung_mask=lung_mask,
                min_radius=args.pseudo_min_radius,
                max_radius=args.pseudo_max_radius,
                valid_erode_iterations=args.pseudo_erode_iterations,
                central_percentile=args.pseudo_central_percentile,
                min_slice_area_percentile=args.pseudo_min_slice_area_percentile,
                position_attempts=args.pseudo_empirical_position_attempts,
                seed_offset=attempt,
                return_disks=True,
            )
        else:
            candidate_matrix, candidate_shepard_detections = sample_pseudo_regions_inside_lung(
                patient_id,
                lung_mask,
                min_regions=args.pseudo_min_regions,
                max_regions=args.pseudo_max_regions,
                min_radius=args.pseudo_min_radius,
                max_radius=args.pseudo_max_radius,
                erode_iterations=args.pseudo_erode_iterations,
                central_percentile=args.pseudo_central_percentile,
                seed_offset=attempt,
                return_disks=True,
            )

        if len(candidate_matrix) == 0:
            print("Rejected: no detector-negative pseudo-region control points.")
            continue

        candidate_matrix, point_labels, z_surface_float, z_surface = fit_surface_grid(
            candidate_matrix,
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
            surface_method=args.surface_method,
            shepard_power=args.shepard_power,
            shepard_detections=candidate_shepard_detections,
        )
        lung_coverage = compute_lung_coverage(z_surface, lung_mask) if lung_mask is not None and np.any(lung_mask) else 0.0
        print(f"Detector-negative lung coverage: {lung_coverage:.3f}")
        if lung_coverage > best_lung_coverage:
            best_lung_coverage = lung_coverage
            best_candidate = (candidate_matrix, point_labels, z_surface_float, z_surface)
        if lung_coverage >= args.min_lung_coverage:
            print("Accepted detector-negative pseudo-region surface.")
            return candidate_matrix, point_labels, z_surface_float, z_surface, "detector_negative_pseudo"
        print("Rejected")

    if best_candidate is not None and best_lung_coverage >= args.min_best_lung_coverage:
        print(
            f"Warning: no detector-negative attempt reached min_lung_coverage={args.min_lung_coverage:.3f}; "
            f"using best attempt with lung coverage {best_lung_coverage:.3f}."
        )
        return (*best_candidate, "detector_negative_pseudo_best")

    print(f"Warning: falling back to mid-slice plane for detector-negative scan {patient_id}.")
    matrix = mid_slice_plane(depth, h, w)
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
        surface_method=args.surface_method,
        shepard_power=args.shepard_power,
        shepard_detections=None,
    )
    return matrix, point_labels, z_surface_float, z_surface, "detector_negative_mid_slice_last_resort"


def save_detector_surfaces(args: argparse.Namespace, dataset: LUNA16NiftiDataset, detections_by_series: dict[str, pd.DataFrame]) -> None:
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    empirical_distribution = (
        load_empirical_nodule_distribution(args.empirical_nodule_distribution_path)
        if args.use_empirical_pseudo_nodules
        else None
    )

    for idx in range(len(dataset)):
        patient_id = "<unknown>"
        try:
            img, _label, patient_id, saved_lung_mask = unpack_sample(dataset[idx])
            current_save_dir = save_path / patient_id
            surface_png = current_save_dir / f"surface_{patient_id}.png"
            required_outputs = [
                surface_png,
                current_save_dir / f"control_points_{patient_id}.npy",
                current_save_dir / f"point_labels_{patient_id}.npy",
            ]
            if args.save_surface_grid:
                required_outputs.extend(
                    [
                        current_save_dir / f"surface_grid_float_{patient_id}.npy",
                        current_save_dir / f"surface_grid_int_{patient_id}.npy",
                    ]
                )
            if args.skip_existing and all(path.exists() for path in required_outputs):
                print(f"Skipping existing complete surface for {patient_id}.")
                continue

            detections = detections_by_series.get(patient_id)
            if detections is None or detections.empty:
                message = f"No detector predictions found for {patient_id}."
                if args.fallback_no_nodule:
                    print(f"Warning: {message} Using detector-negative no-nodule-style fallback.")
                elif args.fallback_mid_slice:
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

            fallback_mode = "detector"
            if detections is None or detections.empty:
                selected = pd.DataFrame(columns=PREDICTION_COLUMNS)
                if args.fallback_no_nodule:
                    if lung_mask is None:
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
                    matrix, point_labels, z_surface_float, z_surface, fallback_mode = detector_negative_surface(
                        args,
                        patient_id,
                        volume_np,
                        lung_mask,
                        h,
                        w,
                        depth,
                        empirical_distribution,
                    )
                else:
                    matrix = mid_slice_plane(depth, h, w)
                    fallback_mode = "mid_slice"
            else:
                selected = detections.head(args.top_k).reset_index(drop=True)
                matrix = detection_control_points(
                    selected,
                    num_contour_points=args.num_contour_points,
                    volume_shape=(depth, h, w),
                )
                shepard_detections = detection_shepard_disks(selected, volume_shape=(depth, h, w))
                print(
                    f"Using {len(matrix)} detector control points "
                    f"({1 + max(0, int(args.num_contour_points))} per detection) from fold {args.fold}; "
                    f"top probability={float(selected['probability'].max()):.4f}."
                )

            if fallback_mode in {"detector", "mid_slice"}:
                if fallback_mode == "mid_slice":
                    shepard_detections = None
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
                    surface_method=args.surface_method,
                    shepard_power=args.shepard_power,
                    shepard_detections=shepard_detections,
                )

            lung_coverage = ""
            if args.report_lung_coverage and lung_mask is not None and np.any(lung_mask):
                lung_coverage = float(compute_lung_coverage(z_surface, lung_mask))
                print(f"Lung coverage: {lung_coverage:.3f}")

            current_save_dir.mkdir(parents=True, exist_ok=True)

            rows, cols = np.indices((h, w))
            img_np = img.numpy()
            output_image = img_np[z_surface, 0, rows, cols]
            output_image = normalize_surface_image(
                output_image,
                lung_window_center=args.lung_window_center,
                lung_window_width=args.lung_window_width,
            )
            Image.fromarray((output_image * 255).astype(np.uint8)).convert("RGB").save(surface_png)

            np.save(current_save_dir / f"control_points_{patient_id}.npy", matrix.astype(np.float32))
            np.save(current_save_dir / f"point_labels_{patient_id}.npy", np.asarray(point_labels))
            if not selected.empty:
                selected.to_csv(current_save_dir / f"detector_top{args.top_k}_{patient_id}.csv", index=False)

            if args.save_surface_grid:
                np.save(current_save_dir / f"surface_grid_float_{patient_id}.npy", z_surface_float.astype(np.float32))
                np.save(current_save_dir / f"surface_grid_int_{patient_id}.npy", z_surface.astype(np.int16))

            manifest_rows.append(
                {
                    "fold": int(args.fold),
                    "split": args.split,
                    "seriesuid": patient_id,
                    "surface_png": str(surface_png),
                    "num_detections": int(len(selected)),
                    "num_control_points": int(len(matrix)),
                    "surface_source": fallback_mode,
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
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.5,
        help="Keep only detector candidates with probability >= this threshold before top-k selection.",
    )
    parser.add_argument("--fallback-mid-slice", action="store_true")
    parser.add_argument(
        "--fallback-no-nodule",
        action="store_true",
        help=(
            "For scans with no detections after thresholding, generate a detector-negative "
            "surface using the same pseudo-region strategy used for no-nodule GT scans."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true", help="Do not overwrite already complete surfaces.")
    parser.add_argument("--save-surface-grid", action="store_true")
    parser.add_argument("--report-lung-coverage", action="store_true")

    parser.add_argument("--rbf-smooth", type=float, default=0.1)
    parser.add_argument(
        "--surface-method",
        choices=("rbf", "shepard"),
        default="rbf",
        help="Surface interpolation method. 'rbf' keeps the existing thin-plate RBF behavior.",
    )
    parser.add_argument("--shepard-power", type=float, default=2.0, help="Inverse-distance power for Shepard surfaces.")
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
    parser.add_argument("--pseudo-min-regions", type=int, default=1)
    parser.add_argument("--pseudo-max-regions", type=int, default=3)
    parser.add_argument("--pseudo-min-radius", type=int, default=8)
    parser.add_argument("--pseudo-max-radius", type=int, default=20)
    parser.add_argument("--pseudo-erode-iterations", type=int, default=2)
    parser.add_argument("--pseudo-central-percentile", type=float, default=70.0)
    parser.add_argument("--pseudo-min-slice-area-percentile", type=float, default=35.0)
    parser.add_argument("--pseudo-empirical-position-attempts", type=int, default=100)
    parser.add_argument("--pseudo-max-attempts", type=int, default=5)
    parser.add_argument("--min-lung-coverage", type=float, default=0.25)
    parser.add_argument("--min-best-lung-coverage", type=float, default=0.10)
    parser.add_argument(
        "--empirical-nodule-distribution-path",
        default="outputs/luna16_saliency_control_point_distribution/empirical_nodule_distribution_from_control_points.npz",
    )
    parser.add_argument("--use-empirical-pseudo-nodules", action=argparse.BooleanOptionalAction, default=True)
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
