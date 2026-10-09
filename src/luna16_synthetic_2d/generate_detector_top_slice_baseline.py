"""Build the axial-slice baseline selected by each scan's top detector prediction.

Detector predictions are read only from their held-out test folds. The image
conversion is shared with the central-axial baseline so the selected z index is
the only difference in the classifier input.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

from .generate_mip_baselines import resize_channel


PREDICTION_COLUMNS = {"seriesuid", "coordZ", "probability"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--splits-dir", default="data/processed/cv_splits")
    parser.add_argument(
        "--pred-root",
        default="outputs/cpmnetv2_luna16_10fold_bf16_guarded_results/normalized_predictions",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 384], metavar=("H", "W"))
    parser.add_argument("--expected-folds", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def detector_fold(path: Path) -> int:
    for part in path.parts:
        match = re.search(r"(?:^|_)fold_?(\d+)$", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Cannot determine detector fold from {path}")


def load_held_out_predictions(pred_root: Path, splits_dir: Path, expected_folds: int) -> pd.DataFrame:
    paths = sorted(pred_root.glob("**/predictions/test_predictions.csv"))
    by_fold: dict[int, Path] = {}
    for path in paths:
        fold = detector_fold(path)
        if fold in by_fold:
            raise ValueError(f"Multiple detector test files for fold {fold}: {by_fold[fold]}, {path}")
        by_fold[fold] = path
    if set(by_fold) != set(range(expected_folds)):
        raise ValueError(f"Expected detector test folds 0..{expected_folds - 1}; found {sorted(by_fold)}")

    predictions = []
    for fold, path in sorted(by_fold.items()):
        frame = pd.read_csv(path)
        missing = PREDICTION_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        frame = frame.dropna(subset=["seriesuid", "coordZ", "probability"]).copy()
        frame["seriesuid"] = frame["seriesuid"].astype(str)
        frame["probability"] = pd.to_numeric(frame["probability"], errors="raise")
        frame["coordZ"] = pd.to_numeric(frame["coordZ"], errors="raise")
        frame["detector_fold"] = fold
        frame["prediction_csv"] = str(path)

        split_path = splits_dir / f"luna16_classification_fold{fold}.csv"
        split_frame = pd.read_csv(split_path)
        held_out = set(split_frame.loc[split_frame["split"] == "test", "seriesuid"].astype(str))
        unexpected = set(frame["seriesuid"]) - held_out
        if unexpected:
            raise ValueError(f"Detector fold {fold} has {len(unexpected)} predictions outside its test split")
        predictions.append(frame)

    combined = pd.concat(predictions, ignore_index=True)
    folds_per_scan = combined.groupby("seriesuid")["detector_fold"].nunique()
    duplicated = folds_per_scan[folds_per_scan != 1]
    if not duplicated.empty:
        raise ValueError(f"Scans predicted by multiple detector folds: {duplicated.index.tolist()[:5]}")
    # Keep the highest-probability detection without an extra score threshold.
    return combined.sort_values("probability", ascending=False, kind="stable").drop_duplicates("seriesuid")


def build_images(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(splits_dir / "luna16_classification_fold0.csv")
    cohort = cohort[cohort["target"].isin([0, 1])].drop_duplicates("seriesuid").copy()
    cohort["seriesuid"] = cohort["seriesuid"].astype(str)
    top = load_held_out_predictions(Path(args.pred_root), splits_dir, args.expected_folds)
    merged = cohort.merge(top, on="seriesuid", how="left", validate="one_to_one")
    if merged["probability"].isna().any():
        missing = merged.loc[merged["probability"].isna(), "seriesuid"].tolist()
        raise ValueError(f"No held-out detector prediction for {len(missing)} scans: {missing[:5]}")

    height, width = (int(v) for v in args.image_size)
    rows = []
    for number, row in enumerate(merged.itertuples(index=False), 1):
        seriesuid = str(row.seriesuid)
        image_path = data_root / row.image_path
        volume = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
        depth = int(volume.shape[0])
        z_index = int(np.clip(np.rint(float(row.coordZ)), 0, depth - 1))
        output_path = output_dir / seriesuid / f"surface_{seriesuid}.png"
        if args.overwrite or not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            channel = resize_channel(volume[z_index], (height, width))
            rgb = np.repeat(channel[:, :, None], 3, axis=2)
            Image.fromarray(rgb, mode="RGB").save(output_path)
        rows.append(
            {
                "seriesuid": seriesuid,
                "label": int(row.target),
                "detector_fold": int(row.detector_fold),
                "detector_probability": float(row.probability),
                "detector_z": float(row.coordZ),
                "selected_z_index": z_index,
                "central_z_index": depth // 2,
                "source_depth": depth,
                "image_path": str(image_path),
                "input_path": str(output_path),
                "prediction_csv": row.prediction_csv,
            }
        )
        if number % 50 == 0:
            print(f"Generated or checked {number}/{len(merged)} detector-top slices", flush=True)

    pd.DataFrame(rows).to_csv(output_dir / "baseline_manifest.csv", index=False)
    metadata = {
        "mode": "detector_top_axial",
        "selection": "highest detector probability from each scan's held-out detector test fold",
        "score_threshold": None,
        "z_index": "clip(round(coordZ), 0, depth - 1)",
        "image_conversion": "same normalize_uint8, bilinear resize, RGB replication as central_axial",
        "image_size": [height, width],
        "n_samples": len(rows),
        "detector_predictions": str(args.pred_root),
    }
    (output_dir / "baseline_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    build_images(parse_args())
