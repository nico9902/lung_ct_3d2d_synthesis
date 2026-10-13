"""Build scan-level classification folds from matched LUNA16 malignancy scores.

The detection fold CSVs define the scan paths and train/val/test membership.
The malignancy CSV is produced by generate_luna_malignancy_csv.py, which matches
official LUNA16 reference nodules to LIDC-IDRI reader ratings. Its inherited
``target`` column is deliberately ignored: classification targets are computed
from ``max_nodule_mean_malignancy``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "seriesuid",
    "lidc_id",
    "image_path",
    "nodule_mask_path",
    "split",
    "target",
    "target_name",
    "max_nodule_mean_malignancy",
    "mean_nodule_malignancy",
    "nodule_count",
    "nodule_annotation_mean_malignancies",
]
TARGETS = {"benign": 0, "malignant": 1, "uncertain": 2}
SPLIT_ORDER = {"test": 0, "train": 1, "val": 2}


def load_scores(scores_csv: Path, annotations_csv: Path) -> pd.DataFrame:
    scores = pd.read_csv(scores_csv, dtype={"seriesuid": str})
    required = {
        "seriesuid",
        "lidc_id",
        "nodule_count",
        "nodule_annotation_mean_malignancies",
        "max_nodule_mean_malignancy",
        "mean_nodule_malignancy",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"{scores_csv} is missing columns: {sorted(missing)}")
    if scores.seriesuid.isna().any() or scores.seriesuid.duplicated().any():
        raise ValueError(f"{scores_csv} must have one non-null row per seriesuid")

    annotations = pd.read_csv(annotations_csv, dtype={"seriesuid": str})
    if "seriesuid" not in annotations.columns:
        raise ValueError(f"{annotations_csv} must contain seriesuid")
    reference_counts = annotations.groupby("seriesuid").size()
    if not set(reference_counts.index).issubset(set(scores.seriesuid)):
        raise ValueError("Some official LUNA16 reference scans are absent from the score CSV")
    actual_counts = scores.seriesuid.map(reference_counts).fillna(0).astype(int)
    if not actual_counts.equals(scores.nodule_count.astype(int)):
        raise ValueError("nodule_count differs from official LUNA16 annotations.csv")

    for row in scores.itertuples(index=False):
        means = json.loads(row.nodule_annotation_mean_malignancies)
        expected = max(means) if means else 0.0
        if not np.isclose(float(row.max_nodule_mean_malignancy), expected):
            raise ValueError(f"Inconsistent maximum nodule mean for {row.seriesuid}")
        if int(row.nodule_count) == 0 and means:
            raise ValueError(f"Scan without LUNA16 references has malignancy ratings: {row.seriesuid}")

    return scores


def build_folds(
    scores_csv: Path,
    annotations_csv: Path,
    detection_splits_dir: Path,
    output_dir: Path,
    num_folds: int = 10,
) -> pd.DataFrame:
    scores = load_scores(scores_csv, annotations_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_columns = [
        "seriesuid",
        "lidc_id",
        "max_nodule_mean_malignancy",
        "mean_nodule_malignancy",
        "nodule_count",
        "nodule_annotation_mean_malignancies",
    ]
    summary_rows = []
    test_membership: dict[str, int] = {}

    for fold in range(num_folds):
        detection_csv = detection_splits_dir / f"luna16_fold{fold}.csv"
        detection = pd.read_csv(detection_csv, dtype={"seriesuid": str})
        required = {"seriesuid", "image_path", "nodule_mask_path", "split", "label"}
        if required - set(detection.columns):
            raise ValueError(f"{detection_csv} is missing columns: {sorted(required - set(detection.columns))}")
        if not set(detection.split).issubset(SPLIT_ORDER):
            raise ValueError(f"{detection_csv} has an unrecognized split")

        per_scan = detection.groupby("seriesuid", sort=False)
        for field in ("image_path", "nodule_mask_path", "split"):
            if per_scan[field].nunique(dropna=False).gt(1).any():
                raise ValueError(f"{detection_csv} has conflicting {field} values within a scan")
        scans = per_scan[["image_path", "nodule_mask_path", "split"]].first().reset_index()
        scans["nodule_mask_path"] = scans["nodule_mask_path"].fillna("")
        counts = detection.loc[detection.label.eq("nodule")].groupby("seriesuid").size()
        scans["detector_nodule_count"] = scans.seriesuid.map(counts).fillna(0).astype(int)

        if set(scans.seriesuid) != set(scores.seriesuid):
            raise ValueError(f"{detection_csv} and {scores_csv} do not cover the same scans")
        scans = scans.merge(scores[score_columns], on="seriesuid", validate="one_to_one")
        if not scans.detector_nodule_count.equals(scans.nodule_count.astype(int)):
            raise ValueError(f"{detection_csv} does not match the LUNA16 reference counts")
        scans = scans.drop(columns="detector_nodule_count")

        score = scans.max_nodule_mean_malignancy.astype(float)
        scans["target_name"] = np.select(
            [score.lt(3), score.gt(3)], ["benign", "malignant"], default="uncertain"
        )
        scans["target"] = scans.target_name.map(TARGETS).astype(int)
        scans["_split_order"] = scans.split.map(SPLIT_ORDER)
        scans = scans.sort_values(["_split_order", "seriesuid"]).drop(columns="_split_order")
        scans = scans[OUTPUT_COLUMNS]

        for seriesuid in scans.loc[scans.split.eq("test"), "seriesuid"]:
            if seriesuid in test_membership:
                raise ValueError(f"Scan is a test case in multiple folds: {seriesuid}")
            test_membership[seriesuid] = fold

        output_csv = output_dir / f"luna16_classification_fold{fold}.csv"
        scans.to_csv(output_csv, index=False)
        for split, group in scans.groupby("split", sort=False):
            summary_rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "samples": len(group),
                    "malignant": int(group.target_name.eq("malignant").sum()),
                    "benign": int(group.target_name.eq("benign").sum()),
                    "uncertain": int(group.target_name.eq("uncertain").sum()),
                    "csv_path": str(output_csv),
                }
            )
        print(f"wrote {output_csv}")

    if set(test_membership) != set(scores.seriesuid):
        raise ValueError("Each scan must be assigned to exactly one test fold")
    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "classification_summary_max_nodule_mean_malignancy.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-csv", type=Path, default=Path("data/raw/LUNA16/luna16_mean_nodule_malignancy.csv")
    )
    parser.add_argument(
        "--annotations-csv", type=Path, default=Path("data/raw/LUNA16/annotations/annotations.csv")
    )
    parser.add_argument("--detection-splits-dir", type=Path, default=Path("data/processed/cv_splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/cv_splits"))
    parser.add_argument("--num-folds", type=int, default=10)
    args = parser.parse_args()
    build_folds(
        scores_csv=args.scores_csv,
        annotations_csv=args.annotations_csv,
        detection_splits_dir=args.detection_splits_dir,
        output_dir=args.output_dir,
        num_folds=args.num_folds,
    )


if __name__ == "__main__":
    main()
