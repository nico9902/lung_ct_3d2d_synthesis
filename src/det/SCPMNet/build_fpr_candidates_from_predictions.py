from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.det.SCPMNet.dataset import prepare_label_dataframe
from src.det.SCPMNet.fp_reduction import ground_truth_by_series, label_candidates
from src.det.SCPMNet.lightning_model import sphere_nms


PREDICTION_COLUMNS = ["seriesuid", "coordZ", "coordY", "coordX", "radius", "probability"]


def fold_index(path: Path) -> int:
    match = re.search(r"fold(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot infer fold index from {path}")
    return int(match.group(1))


def load_prediction_pool(prediction_root: Path, fold_glob: str, prediction_name: str) -> pd.DataFrame:
    rows = []
    fold_dirs = sorted(prediction_root.glob(fold_glob), key=fold_index)
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories matched {prediction_root / fold_glob}")

    for fold_dir in fold_dirs:
        prediction_path = fold_dir / "predictions" / prediction_name
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction CSV: {prediction_path}")
        frame = pd.read_csv(prediction_path)
        missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{prediction_path} is missing columns: {missing}")
        frame = frame[PREDICTION_COLUMNS].copy()
        frame["source_fold"] = fold_index(fold_dir)
        rows.append(frame)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[*PREDICTION_COLUMNS, "source_fold"])


def seriesuids_for_split(csv_path: Path, split: str, data_root: Path, skip_missing_images: bool) -> set[str]:
    labels = prepare_label_dataframe(str(csv_path), split, data_root, skip_missing_images=skip_missing_images)
    return set(labels["seriesuid"].astype(str).unique())


def top_candidates_by_series(predictions: pd.DataFrame, top_k: int, nms_threshold: float) -> pd.DataFrame:
    output_rows = []
    for seriesuid, group in predictions.groupby("seriesuid", sort=False):
        detections = torch.as_tensor(
            group[["coordZ", "coordY", "coordX", "radius", "probability"]].to_numpy(),
            dtype=torch.float32,
        )
        detections = sphere_nms(detections, float(nms_threshold), int(top_k))
        if len(detections) == 0:
            continue
        source_fold = int(group["source_fold"].iloc[0]) if "source_fold" in group.columns else -1
        for rank, (z, y, x, radius, score) in enumerate(detections.tolist(), start=1):
            output_rows.append(
                {
                    "seriesuid": str(seriesuid),
                    "coordZ": z,
                    "coordY": y,
                    "coordX": x,
                    "radius": radius,
                    "probability": score,
                    "candidate_rank": rank,
                    "source_fold": source_fold,
                }
            )
    return pd.DataFrame(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build FPR candidate CSVs from existing detector prediction CSVs. "
            "This avoids rerunning detector inference when fold test_predictions.csv files already exist."
        )
    )
    parser.add_argument("--prediction-root", type=Path, default=Path("outputs/scpmnet_luna16_10fold"))
    parser.add_argument("--fold-glob", default="scpmnet_paper_luna16_fold*")
    parser.add_argument("--prediction-name", default="test_predictions.csv")
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--top-candidates-per-volume", type=int, default=100)
    parser.add_argument("--nms-threshold", type=float, default=0.05)
    parser.add_argument("--label-candidates", action="store_true")
    parser.add_argument("--ignore-margin", type=float, default=2.0)
    parser.add_argument("--keep-missing-images", action="store_true")
    args = parser.parse_args()

    prediction_pool = load_prediction_pool(args.prediction_root, args.fold_glob, args.prediction_name)
    wanted_seriesuids = seriesuids_for_split(
        args.csv_path,
        args.split,
        args.data_root,
        skip_missing_images=not args.keep_missing_images,
    )
    filtered = prediction_pool[prediction_pool["seriesuid"].astype(str).isin(wanted_seriesuids)].copy()
    candidates = top_candidates_by_series(filtered, args.top_candidates_per_volume, args.nms_threshold)
    if args.label_candidates:
        gt = ground_truth_by_series(
            args.csv_path,
            args.split,
            args.data_root,
            skip_missing_images=not args.keep_missing_images,
        )
        candidates = label_candidates(candidates, gt, ignore_margin=args.ignore_margin)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(candidates)} candidates to {args.output_csv}")
    print(f"Split seriesuids: {len(wanted_seriesuids)} | with detections: {candidates['seriesuid'].nunique() if not candidates.empty else 0}")
    if "label" in candidates.columns and not candidates.empty:
        usable = candidates[~candidates["ignore"].astype(bool)] if "ignore" in candidates.columns else candidates
        print(usable["label"].value_counts().rename(index={0: "negative", 1: "positive"}).to_string())


if __name__ == "__main__":
    main()
