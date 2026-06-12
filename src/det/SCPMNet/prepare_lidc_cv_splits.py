from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def assign_folds(seriesuids: list[str], num_folds: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sorted(seriesuids), dtype=object)
    rng.shuffle(shuffled)
    return {str(seriesuid): int(i % num_folds) for i, seriesuid in enumerate(shuffled)}


def assign_stratified_folds(seriesuids: list[str], positive_series: set[str], num_folds: int, seed: int) -> dict[str, int]:
    fold_by_series: dict[str, int] = {}
    positives = [seriesuid for seriesuid in seriesuids if seriesuid in positive_series]
    negatives = [seriesuid for seriesuid in seriesuids if seriesuid not in positive_series]

    for offset, group in enumerate((positives, negatives)):
        group_folds = assign_folds(group, num_folds=num_folds, seed=seed + offset)
        fold_by_series.update(group_folds)

    return fold_by_series


def full_lidc_series(all_samples_csv: Path | None) -> list[str] | None:
    if all_samples_csv is None:
        return None
    samples = pd.read_csv(all_samples_csv)
    if "lidc_id" in samples.columns:
        ids = samples["lidc_id"]
    elif "patient_id" in samples.columns:
        ids = samples["patient_id"]
    elif "seriesuid" in samples.columns:
        ids = samples["seriesuid"]
    else:
        raise ValueError(f"{all_samples_csv} must contain lidc_id, patient_id, or seriesuid.")
    return sorted(ids.astype(str).unique())


def add_negative_rows(labels: pd.DataFrame, seriesuids: list[str]) -> pd.DataFrame:
    labels = labels.copy()
    labels["seriesuid"] = labels["seriesuid"].astype(str)
    existing = set(labels["seriesuid"].unique())
    missing = [seriesuid for seriesuid in seriesuids if seriesuid not in existing]
    if not missing:
        return labels

    negative_rows = []
    template_cols = list(labels.columns)
    for seriesuid in missing:
        row = {col: None for col in template_cols}
        row["seriesuid"] = seriesuid
        row["label"] = "negative"
        negative_rows.append(row)

    negative_df = pd.DataFrame(negative_rows, columns=template_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return pd.concat([labels, negative_df], ignore_index=True)


def add_paths(df: pd.DataFrame, preprocessed_root: Path) -> pd.DataFrame:
    df = df.copy()

    def image_path(seriesuid: str) -> str:
        path = preprocessed_root / seriesuid / f"{seriesuid}_volume.nii.gz"
        return relpath(path, preprocessed_root)

    def mask_path(seriesuid: str) -> str:
        path = preprocessed_root / seriesuid / f"{seriesuid}_nodule_mask.nii.gz"
        return relpath(path, preprocessed_root) if path.exists() else ""

    series = df["seriesuid"].astype(str)
    if "image_path" not in df.columns:
        df.insert(1, "image_path", series.map(image_path))
    else:
        df["image_path"] = df["image_path"].fillna(series.map(image_path))

    if "nodule_mask_path" not in df.columns:
        insert_at = min(2, len(df.columns))
        df.insert(insert_at, "nodule_mask_path", series.map(mask_path))
    else:
        df["nodule_mask_path"] = df["nodule_mask_path"].fillna(series.map(mask_path))

    return df


def build_fold_csvs(
    preprocessed_root: Path,
    labels_csv: Path,
    all_samples_csv: Path | None,
    output_dir: Path,
    num_folds: int,
    val_offset: int,
    seed: int,
) -> None:
    labels = pd.read_csv(labels_csv)
    if "seriesuid" not in labels.columns:
        raise ValueError(f"{labels_csv} must contain a seriesuid column.")

    labels["seriesuid"] = labels["seriesuid"].astype(str)
    full_seriesuids = full_lidc_series(all_samples_csv)
    if full_seriesuids is not None:
        labels = add_negative_rows(labels, full_seriesuids)
    labels = add_paths(labels, preprocessed_root)

    seriesuids = full_seriesuids if full_seriesuids is not None else sorted(labels["seriesuid"].unique())
    if not seriesuids:
        raise ValueError(f"No LIDC series found in {labels_csv}")
    if num_folds < 2:
        raise ValueError("--num-folds must be at least 2.")
    if num_folds > len(seriesuids):
        raise ValueError(f"--num-folds={num_folds} is larger than the number of series ({len(seriesuids)}).")

    positive_mask = labels["label"].astype(str).str.lower().isin(("nodule", "1", "true", "positive"))
    positive_series = set(labels.loc[positive_mask, "seriesuid"].astype(str).unique())
    fold_by_series = assign_stratified_folds(seriesuids, positive_series, num_folds=num_folds, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for test_fold in range(num_folds):
        val_fold = (test_fold + val_offset) % num_folds
        fold_df = labels.copy()
        fold_df["split"] = fold_df["seriesuid"].map(
            lambda seriesuid: "test"
            if fold_by_series[seriesuid] == test_fold
            else "val"
            if fold_by_series[seriesuid] == val_fold
            else "train"
        )

        ordered_cols = [
            col
            for col in (
                "seriesuid",
                "image_path",
                "nodule_mask_path",
                "split",
                "x",
                "y",
                "z",
                "w",
                "h",
                "d",
        "diameter_mm",
                "label",
                "target",
                "nodule_count",
            )
            if col in fold_df.columns
        ]
        remaining_cols = [col for col in fold_df.columns if col not in ordered_cols]
        fold_df = fold_df[ordered_cols + remaining_cols]

        fold_path = output_dir / f"lidc_fold{test_fold}.csv"
        fold_df.to_csv(fold_path, index=False)

        for split, rows in fold_df.groupby("split", sort=False):
            image_exists = rows.groupby("seriesuid")["image_path"].first().map(lambda p: (preprocessed_root / str(p)).exists())
            summary_rows.append(
                {
                    "fold": test_fold,
                    "test_fold": test_fold,
                    "val_fold": val_fold,
                    "split": split,
                    "rows": int(len(rows)),
                    "series": int(rows["seriesuid"].nunique()),
                    "nodules": int(rows["label"].astype(str).str.lower().isin(("nodule", "1", "true", "positive")).sum())
                    if "label" in rows.columns
                    else int(len(rows)),
                    "series_with_existing_images": int(image_exists.sum()),
                    "csv_path": str(fold_path),
                }
            )
        print(f"wrote {fold_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SCPMNet 10-fold CSV files for preprocessed LIDC.")
    parser.add_argument("--preprocessed-root", default="data/lidc_process", help="Root containing LIDC-IDRI-* folders.")
    parser.add_argument("--labels-csv", default="data/lidc_process/lidc_labels.csv", help="LIDC labels CSV.")
    parser.add_argument(
        "--all-samples-csv",
        default="data/LIDC-IDRI_seriesuid_mapping_all_1010.csv",
        help="CSV defining the full LIDC sample universe. Use an empty string to split only labeled positives.",
    )
    parser.add_argument("--output-dir", default="data/lidc_process/cv_splits", help="Directory for lidc_fold*.csv files.")
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument(
        "--val-offset",
        type=int,
        default=1,
        help="Validation fold is (test_fold + val_offset) %% num_folds. Default: 1.",
    )
    parser.add_argument("--seed", type=int, default=233)
    args = parser.parse_args()

    build_fold_csvs(
        preprocessed_root=Path(args.preprocessed_root),
        labels_csv=Path(args.labels_csv),
        all_samples_csv=Path(args.all_samples_csv) if args.all_samples_csv else None,
        output_dir=Path(args.output_dir),
        num_folds=args.num_folds,
        val_offset=args.val_offset,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
