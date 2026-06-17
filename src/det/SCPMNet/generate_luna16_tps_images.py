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


def detection_points(detections: pd.DataFrame) -> np.ndarray:
    return detections[["coordZ", "coordY", "coordX"]].to_numpy(dtype=np.float32)


def thin_plate_surface(
    detections: pd.DataFrame,
    shape: tuple[int, int, int],
    smooth: float,
    use_boundary_anchors: bool,
) -> np.ndarray:
    depth, height, width = shape
    points = detection_points(detections)
    if len(points) == 0:
        raise ValueError("Cannot build a TPS surface without detections.")

    mean_z = float(np.clip(np.mean(points[:, 0]), 0.0, depth - 1.0))
    if len(points) == 1:
        return np.full((height, width), mean_z, dtype=np.float32)

    if use_boundary_anchors:
        points = np.vstack([points, boundary_anchors(shape, mean_z)])

    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    try:
        rbf = Rbf(points[:, 2], points[:, 1], points[:, 0], function="thin_plate", smooth=float(smooth))
    except np.linalg.LinAlgError:
        rbf = Rbf(points[:, 2], points[:, 1], points[:, 0], function="thin_plate", smooth=max(float(smooth), 1e-3))
    z_surface = rbf(x_grid, y_grid).astype(np.float32)
    return np.clip(z_surface, 0.0, float(depth - 1))


def sample_surface(volume: np.ndarray, z_surface: np.ndarray) -> np.ndarray:
    height, width = z_surface.shape
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    coords = np.vstack([z_surface.ravel(), y_grid.ravel(), x_grid.ravel()])
    image = map_coordinates(volume, coords, order=1, mode="nearest").reshape(height, width)
    return image.astype(np.float32)


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
    by_series = top_detections(predictions, args.top_k)
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
        z_surface = thin_plate_surface(
            detections=detections,
            shape=tuple(int(v) for v in volume.shape),
            smooth=args.smooth,
            use_boundary_anchors=not args.no_boundary_anchors,
        )
        synthetic = sample_surface(volume, z_surface)
        image_u8 = to_uint8(synthetic)

        stem = output_stem(seriesuid)
        image_out = out_dir / f"{stem}_tps_top{args.top_k}.png"
        surface_out = out_dir / f"{stem}_tps_z_surface.npy"
        Image.fromarray(image_u8, mode="L").save(image_out)
        if args.save_surfaces:
            np.save(surface_out, z_surface.astype(np.float32))

        overlay_out = ""
        if not args.no_overlays:
            overlay_path = out_dir / f"{stem}_tps_top{args.top_k}_overlay.png"
            save_overlay(image_u8, detections, overlay_path)
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
                "num_detections": int(len(detections)),
                "top_probability": float(detections["probability"].max()),
                "mean_detection_z": float(detections["coordZ"].mean()),
            }
        )

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"[fold {fold}] wrote {len(manifest_rows)} TPS images to {out_dir}")
    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LUNA16 SCPMNet synthetic 2D CT images through top detections using TPS/RBF interpolation."
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
    parser.add_argument("--clip", type=float, nargs=2, default=(-1000.0, 400.0))
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--save-surfaces", action="store_true")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--no-boundary-anchors", action="store_true")
    parser.add_argument("--keep-missing-images", action="store_true")
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
    print(f"Wrote {len(all_rows)} total TPS images.")


if __name__ == "__main__":
    main()
