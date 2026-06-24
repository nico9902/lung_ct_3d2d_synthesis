from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.interpolate import Rbf
from scipy.ndimage import map_coordinates

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.det.SCPMNet.dataset import load_volume, normalize_ct, prepare_label_dataframe


PREDICTION_COLUMNS = ("seriesuid", "coordZ", "coordY", "coordX", "radius", "probability")


def unique_positive_ints(values: list[int]) -> list[int]:
    out = []
    for value in values:
        value = int(value)
        if value > 0 and value not in out:
            out.append(value)
    return out


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


def prediction_csv_for_fold(pred_root: Path, fold: int, prediction_name: str) -> Path | None:
    fold_dir = find_fold_dir(pred_root, fold)
    if fold_dir is None:
        return None
    candidates = [
        fold_dir / "predictions" / prediction_name,
        fold_dir / prediction_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def top_detections(predictions: pd.DataFrame, top_k: int) -> dict[str, pd.DataFrame]:
    if predictions.empty:
        return {}
    missing = [col for col in PREDICTION_COLUMNS if col not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {missing}")
    predictions = predictions.dropna(subset=["coordZ", "coordY", "coordX", "probability"]).copy()
    predictions["seriesuid"] = predictions["seriesuid"].astype(str)
    predictions["probability"] = predictions["probability"].astype(float)
    out = {}
    for seriesuid, rows in predictions.groupby("seriesuid", sort=False):
        out[str(seriesuid)] = rows.sort_values("probability", ascending=False).head(top_k).reset_index(drop=True)
    return out


def candidate_detections(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if predictions.empty:
        return {}
    missing = [col for col in PREDICTION_COLUMNS if col not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {missing}")
    predictions = predictions.dropna(subset=["coordZ", "coordY", "coordX", "probability"]).copy()
    predictions["seriesuid"] = predictions["seriesuid"].astype(str)
    predictions["probability"] = predictions["probability"].astype(float)
    out = {}
    for seriesuid, rows in predictions.groupby("seriesuid", sort=False):
        out[str(seriesuid)] = rows.sort_values("probability", ascending=False).reset_index(drop=True)
    return out


def filter_boundary_detections(
    detections: pd.DataFrame,
    shape: tuple[int, int, int],
    boundary_margin: float,
    min_probability: float | None,
) -> tuple[pd.DataFrame, int, int]:
    depth, height, width = shape
    if detections.empty:
        return detections.copy(), 0, 0

    filtered = detections.copy()
    if min_probability is not None:
        filtered = filtered[filtered["probability"].astype(float) >= float(min_probability)]

    if boundary_margin > 0:
        margin = float(boundary_margin)
        radius = filtered["radius"].fillna(0.0).astype(float).clip(lower=0.0)
        x_margin = np.maximum(margin, radius)
        y_margin = np.maximum(margin, radius)
        z_margin = np.maximum(margin, radius)
        inside = (
            (filtered["coordX"].astype(float) >= x_margin)
            & (filtered["coordX"].astype(float) <= (width - 1) - x_margin)
            & (filtered["coordY"].astype(float) >= y_margin)
            & (filtered["coordY"].astype(float) <= (height - 1) - y_margin)
            & (filtered["coordZ"].astype(float) >= z_margin)
            & (filtered["coordZ"].astype(float) <= (depth - 1) - z_margin)
        )
        filtered = filtered[inside]

    filtered = filtered.sort_values("probability", ascending=False).reset_index(drop=True)
    return filtered, int(len(detections) - len(filtered)), int(len(detections))


def boundary_anchors(shape: tuple[int, int, int], z_value: float) -> np.ndarray:
    _, height, width = shape
    points = []
    for y in (0.0, (height - 1) / 2.0, float(height - 1)):
        points.append([z_value, y, 0.0])
        points.append([z_value, y, float(width - 1)])
    for x in ((width - 1) / 2.0,):
        points.append([z_value, 0.0, x])
        points.append([z_value, float(height - 1), x])
    return np.asarray(points, dtype=np.float32)


def grid_anchors(shape: tuple[int, int, int], z_value: float, grid_size: int) -> np.ndarray:
    if grid_size <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    _, height, width = shape
    ys = np.linspace(0.0, float(height - 1), int(grid_size) + 2, dtype=np.float32)[1:-1]
    xs = np.linspace(0.0, float(width - 1), int(grid_size) + 2, dtype=np.float32)[1:-1]
    points = [[z_value, float(y), float(x)] for y in ys for x in xs]
    return np.asarray(points, dtype=np.float32)


def detection_points(detections: pd.DataFrame) -> np.ndarray:
    return detections[["coordZ", "coordY", "coordX"]].to_numpy(dtype=np.float32)


def thin_plate_surface(
    detections: pd.DataFrame,
    shape: tuple[int, int, int],
    smooth: float,
    use_boundary_anchors: bool,
    anchor_grid_size: int,
    surface_clip_margin: float | None,
) -> np.ndarray:
    depth, height, width = shape
    points = detection_points(detections)
    if len(points) == 0:
        raise ValueError("Cannot build a TPS surface without detections.")

    mean_z = float(np.clip(np.mean(points[:, 0]), 0.0, depth - 1.0))
    if surface_clip_margin is None or surface_clip_margin < 0:
        clip_min = 0.0
        clip_max = float(depth - 1)
    else:
        clip_min = float(np.clip(np.min(points[:, 0]) - surface_clip_margin, 0.0, depth - 1.0))
        clip_max = float(np.clip(np.max(points[:, 0]) + surface_clip_margin, 0.0, depth - 1.0))
    if len(points) == 1:
        return np.full((height, width), mean_z, dtype=np.float32)

    if use_boundary_anchors:
        points = np.vstack([points, boundary_anchors(shape, mean_z)])
    grid_points = grid_anchors(shape, mean_z, anchor_grid_size)
    if len(grid_points):
        points = np.vstack([points, grid_points])

    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    try:
        rbf = Rbf(points[:, 2], points[:, 1], points[:, 0], function="thin_plate", smooth=float(smooth))
    except np.linalg.LinAlgError:
        rbf = Rbf(points[:, 2], points[:, 1], points[:, 0], function="thin_plate", smooth=max(float(smooth), 1e-3))
    z_surface = rbf(x_grid, y_grid).astype(np.float32)
    return np.clip(z_surface, clip_min, clip_max)


def sample_surface(volume: np.ndarray, z_surface: np.ndarray) -> np.ndarray:
    height, width = z_surface.shape
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    coords = np.vstack([z_surface.ravel(), y_grid.ravel(), x_grid.ravel()])
    image = map_coordinates(volume, coords, order=1, mode="nearest").reshape(height, width)
    return image.astype(np.float32)


def surface_qc_metrics(detections: pd.DataFrame, z_surface: np.ndarray) -> dict[str, float]:
    gy, gx = np.gradient(z_surface)
    slope = np.sqrt(gx * gx + gy * gy)
    gyy, _ = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    curvature = np.sqrt(gxx * gxx + gyy * gyy + 2.0 * gxy * gxy)
    detection_z_spread = float(detections["coordZ"].max() - detections["coordZ"].min()) if len(detections) else 0.0
    return {
        "detection_z_spread": detection_z_spread,
        "surface_z_range": float(z_surface.max() - z_surface.min()),
        "surface_z_std": float(z_surface.std()),
        "surface_slope_p99": float(np.percentile(slope, 99)),
        "surface_slope_max": float(slope.max()),
        "surface_curv_p99": float(np.percentile(curvature, 99)),
        "surface_curv_max": float(curvature.max()),
    }


def qc_grade(metrics: dict[str, float], args: argparse.Namespace) -> tuple[str, float]:
    good = (
        metrics["surface_z_range"] <= args.qc_good_z_range
        and metrics["surface_slope_p99"] <= args.qc_good_slope_p99
        and metrics["surface_curv_p99"] <= args.qc_good_curv_p99
    )
    if good:
        return "good", float(args.good_weight)

    medium = (
        metrics["surface_z_range"] <= args.qc_medium_z_range
        and metrics["surface_slope_p99"] <= args.qc_medium_slope_p99
        and metrics["surface_curv_p99"] <= args.qc_medium_curv_p99
    )
    if medium:
        return "medium", float(args.medium_weight)
    return "bad", float(args.bad_weight)


def qc_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    return (
        metrics["surface_z_range"] / max(args.qc_good_z_range, 1e-6)
        + metrics["surface_slope_p99"] / max(args.qc_good_slope_p99, 1e-6)
        + metrics["surface_curv_p99"] / max(args.qc_good_curv_p99, 1e-6)
    )


def generate_surface_with_fallbacks(
    detections: pd.DataFrame,
    shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float], str, float, int]:
    fallback_top_ks = unique_positive_ints([args.top_k, *args.fallback_top_k])
    attempts = []
    for top_k in fallback_top_ks:
        if detections.empty:
            break
        selected = detections.head(min(top_k, len(detections))).reset_index(drop=True)
        z_surface = thin_plate_surface(
            detections=selected,
            shape=shape,
            smooth=args.smooth,
            use_boundary_anchors=not args.no_boundary_anchors,
            anchor_grid_size=args.anchor_grid_size,
            surface_clip_margin=args.surface_clip_margin,
        )
        metrics = surface_qc_metrics(selected, z_surface)
        quality, sample_weight = qc_grade(metrics, args)
        attempts.append((selected, z_surface, metrics, quality, sample_weight, top_k, qc_score(metrics, args)))
        if quality == "good":
            return selected, z_surface, metrics, quality, sample_weight, top_k

    if not attempts:
        raise ValueError("Cannot generate a TPS surface without usable detections.")

    selected, z_surface, metrics, quality, sample_weight, top_k, _ = min(attempts, key=lambda item: item[-1])
    return selected, z_surface, metrics, quality, sample_weight, top_k


def to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, -1.0, 1.0)
    return ((image + 1.0) * 127.5).round().astype(np.uint8)


def save_overlay(image_u8: np.ndarray, detections: pd.DataFrame, path: Path) -> None:
    rgb = Image.fromarray(image_u8, mode="L").convert("RGB")
    draw = ImageDraw.Draw(rgb)
    for _, row in detections.iterrows():
        x = float(row["coordX"])
        y = float(row["coordY"])
        radius = max(float(row.get("radius", 4.0)), 3.0)
        box = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(box, outline=(255, 48, 48), width=2)
    rgb.save(path)


def output_stem(seriesuid: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in seriesuid)


def generate_for_fold(args: argparse.Namespace, fold: int) -> list[dict[str, object]]:
    prediction_csv = prediction_csv_for_fold(Path(args.pred_root), fold, args.prediction_name)
    if prediction_csv is None:
        print(f"[fold {fold}] no prediction CSV found; skipping")
        return []

    split_csv = Path(args.split_dir) / f"luna16_fold{fold}.csv"
    if not split_csv.exists():
        print(f"[fold {fold}] split CSV not found at {split_csv}; skipping")
        return []

    predictions = pd.read_csv(prediction_csv)
    by_series = candidate_detections(predictions)
    if args.max_scans is not None:
        by_series = dict(list(by_series.items())[: args.max_scans])

    labels = prepare_label_dataframe(str(split_csv), args.split, Path(args.data_root), skip_missing_images=not args.keep_missing_images)
    image_paths = {str(seriesuid): Path(rows.iloc[0]["_resolved_image_path"]) for seriesuid, rows in labels.groupby("seriesuid", sort=False)}
    out_dir = Path(args.output_dir) / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for seriesuid, detections in by_series.items():
        image_path = image_paths.get(seriesuid)
        if image_path is None or not image_path.exists():
            print(f"[fold {fold}] missing image for {seriesuid}; skipping")
            continue

        volume = normalize_ct(load_volume(image_path), tuple(args.clip))
        shape = tuple(int(v) for v in volume.shape)
        filtered_detections, filtered_out, num_candidates = filter_boundary_detections(
            detections=detections,
            shape=shape,
            boundary_margin=args.boundary_margin,
            min_probability=args.min_probability,
        )
        used_unfiltered_fallback = False
        if filtered_detections.empty:
            filtered_detections = detections.head(1).reset_index(drop=True)
            used_unfiltered_fallback = True

        selected_detections, z_surface, qc_metrics, qc_quality, sample_weight, effective_top_k = generate_surface_with_fallbacks(
            detections=filtered_detections,
            shape=shape,
            args=args,
        )
        if used_unfiltered_fallback:
            qc_quality = "boundary_fallback"
            sample_weight = min(sample_weight, float(args.bad_weight))

        synthetic = sample_surface(volume, z_surface)

        stem = output_stem(seriesuid)
        image_out = out_dir / f"{stem}_tps_top{args.top_k}.npy"
        surface_out = out_dir / f"{stem}_tps_z_surface.npy"
        np.save(image_out, synthetic.astype(np.float32))
        if args.save_surfaces:
            np.save(surface_out, z_surface.astype(np.float32))

        overlay_out = ""
        if args.save_overlays:
            image_u8 = to_uint8(synthetic)
            overlay_path = out_dir / f"{stem}_tps_top{args.top_k}_overlay.png"
            save_overlay(image_u8, selected_detections, overlay_path)
            overlay_out = str(overlay_path)

        manifest_rows.append(
            {
                "fold": fold,
                "seriesuid": seriesuid,
                "image_path": str(image_path),
                "prediction_csv": str(prediction_csv),
                "synthetic_image": str(image_out),
                "overlay_image": overlay_out,
                "z_surface": str(surface_out) if args.save_surfaces else "",
                "num_candidates": int(num_candidates),
                "num_boundary_filtered": int(filtered_out),
                "num_detections": int(len(selected_detections)),
                "requested_top_k": int(args.top_k),
                "effective_top_k": int(effective_top_k),
                "anchor_grid_size": int(args.anchor_grid_size),
                "surface_clip_margin": float(args.surface_clip_margin) if args.surface_clip_margin is not None else "",
                "boundary_margin": float(args.boundary_margin),
                "used_unfiltered_boundary_fallback": bool(used_unfiltered_fallback),
                "qc_quality": qc_quality,
                "sample_weight": float(sample_weight),
                "top_probability": float(selected_detections["probability"].max()),
                "mean_detection_z": float(selected_detections["coordZ"].mean()),
                "min_detection_z": float(selected_detections["coordZ"].min()),
                "max_detection_z": float(selected_detections["coordZ"].max()),
                **qc_metrics,
            }
        )

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"[fold {fold}] wrote {len(manifest_rows)} TPS image arrays to {out_dir}")
    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LUNA16 SCPMNet synthetic 2D CT image arrays through top detections using TPS/RBF interpolation."
    )
    parser.add_argument("--pred-root", default="outputs/scpmnet_luna16_10fold")
    parser.add_argument("--prediction-name", default="test_predictions.csv")
    parser.add_argument("--split-dir", default="data/LUNA16_preprocessed/cv_splits")
    parser.add_argument("--data-root", default="data/LUNA16_preprocessed")
    parser.add_argument("--output-dir", default="outputs/scpmnet_luna16_10fold_tps_images")
    parser.add_argument("--split", default="test")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--smooth", type=float, default=0.0)
    parser.add_argument(
        "--anchor-grid-size",
        type=int,
        default=5,
        help="Number of internal TPS anchor points per axis. 5 means a 5x5 interior grid at mean detection z.",
    )
    parser.add_argument(
        "--surface-clip-margin",
        type=float,
        default=40.0,
        help="Limit TPS z values to [min_detection_z-margin, max_detection_z+margin]. Use a negative value to disable.",
    )
    parser.add_argument("--clip", type=float, nargs=2, default=(-1000.0, 400.0))
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--save-surfaces", action="store_true")
    parser.add_argument("--save-overlays", action="store_true", help="Also write PNG overlays for visual inspection.")
    parser.add_argument("--no-overlays", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-boundary-anchors", action="store_true")
    parser.add_argument("--keep-missing-images", action="store_true")
    parser.add_argument(
        "--fallback-top-k",
        type=int,
        nargs="*",
        default=[3, 1],
        help="Fallback top-k values to try when the requested top-k surface fails QC.",
    )
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=8.0,
        help="Discard candidate detections whose center/radius is within this many voxels of any volume boundary.",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=None,
        help="Optional probability threshold before boundary filtering and top-k fallback.",
    )
    parser.add_argument("--qc-good-z-range", type=float, default=180.0)
    parser.add_argument("--qc-good-slope-p99", type=float, default=10.0)
    parser.add_argument("--qc-good-curv-p99", type=float, default=2.5)
    parser.add_argument("--qc-medium-z-range", type=float, default=240.0)
    parser.add_argument("--qc-medium-slope-p99", type=float, default=16.0)
    parser.add_argument("--qc-medium-curv-p99", type=float, default=5.0)
    parser.add_argument("--good-weight", type=float, default=1.0)
    parser.add_argument("--medium-weight", type=float, default=0.5)
    parser.add_argument("--bad-weight", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows = []
    for fold in args.folds:
        all_rows.extend(generate_for_fold(args, int(fold)))
    if all_rows:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"Wrote {len(all_rows)} total TPS image arrays.")


if __name__ == "__main__":
    main()
