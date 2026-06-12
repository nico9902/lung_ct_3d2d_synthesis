from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.det.SCPMNet.prepare_luna16_cv_splits import annotation_rows, find_series_dirs


def assign_split(
    seriesuids: list[str],
    positive_series: set[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    split_by_series: dict[str, str] = {}

    def split_group(group: list[str]) -> None:
        shuffled = np.asarray(sorted(group), dtype=object)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        n_test = min(max(n_test, 1 if n > 2 and test_fraction > 0 else 0), n)
        n_val = min(max(n_val, 1 if n - n_test > 2 and val_fraction > 0 else 0), n - n_test)
        for seriesuid in shuffled[:n_test]:
            split_by_series[str(seriesuid)] = "test"
        for seriesuid in shuffled[n_test : n_test + n_val]:
            split_by_series[str(seriesuid)] = "val"
        for seriesuid in shuffled[n_test + n_val :]:
            split_by_series[str(seriesuid)] = "train"

    positives = [seriesuid for seriesuid in seriesuids if seriesuid in positive_series]
    negatives = [seriesuid for seriesuid in seriesuids if seriesuid not in positive_series]
    split_group(positives)
    split_group(negatives)
    return split_by_series


def build_holdout_csv(
    preprocessed_root: Path,
    annotations_csv: Path,
    output_csv: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> None:
    annotations = pd.read_csv(annotations_csv)
    series = find_series_dirs(preprocessed_root)
    if not series:
        raise ValueError(f"No subset*/<seriesuid> volume folders found under {preprocessed_root}")

    positive_series = set(annotations["seriesuid"].astype(str).unique())
    split_by_series = assign_split(
        seriesuids=list(series),
        positive_series=positive_series,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    rows = []
    for seriesuid, (_, series_dir, volume_path) in sorted(series.items()):
        split = split_by_series[seriesuid]
        mask_path = series_dir / f"{seriesuid}_nodule_mask.nii.gz"
        rows.extend(
            annotation_rows(
                seriesuid=seriesuid,
                split=split,
                image_path=volume_path,
                mask_path=mask_path,
                data_root=preprocessed_root,
                annotations=annotations,
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    summary = (
        df.groupby("split")
        .agg(
            rows=("seriesuid", "size"),
            series=("seriesuid", "nunique"),
            nodules=("label", lambda s: int((s == "nodule").sum())),
        )
        .reset_index()
    )
    summary_path = output_csv.with_name(f"{output_csv.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"wrote {output_csv}")
    print(f"wrote {summary_path}")
    print(summary.to_string(index=False))


def build_holdout_csv_from_source(
    source_csv: Path,
    output_csv: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> None:
    df = pd.read_csv(source_csv)
    if "seriesuid" not in df.columns:
        raise ValueError(f"{source_csv} must contain a seriesuid column.")
    if "label" in df.columns:
        positive_mask = df["label"].astype(str).str.lower().isin(("nodule", "1", "true", "positive"))
        positive_series = set(df.loc[positive_mask, "seriesuid"].astype(str).unique())
    else:
        positive_series = set()

    seriesuids = sorted(df["seriesuid"].astype(str).unique())
    split_by_series = assign_split(
        seriesuids=seriesuids,
        positive_series=positive_series,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    df["split"] = df["seriesuid"].astype(str).map(split_by_series)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    summary = (
        df.groupby("split")
        .agg(
            rows=("seriesuid", "size"),
            series=("seriesuid", "nunique"),
            nodules=("label", lambda s: int(s.astype(str).str.lower().isin(("nodule", "1", "true", "positive")).sum()))
            if "label" in df.columns
            else ("seriesuid", "size"),
        )
        .reset_index()
    )
    summary_path = output_csv.with_name(f"{output_csv.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"read {source_csv}")
    print(f"wrote {output_csv}")
    print(f"wrote {summary_path}")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one random train/val/test hold-out CSV for LUNA16 SCPMNet.")
    parser.add_argument(
        "--source-csv",
        default=None,
        help="Prepared LUNA16 CSV to reshuffle directly, e.g. LUNA16_preprocessed/cv_splits/luna16_fold0.csv.",
    )
    parser.add_argument("--preprocessed-root", default=None, help="Root containing subset0 ... subset9 folders.")
    parser.add_argument("--annotations-csv", default=None, help="Official LUNA16 annotations.csv.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=233)
    args = parser.parse_args()

    if args.val_fraction < 0 or args.test_fraction < 0 or args.val_fraction + args.test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to less than 1.")

    if args.source_csv:
        build_holdout_csv_from_source(
            source_csv=Path(args.source_csv),
            output_csv=Path(args.output_csv),
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    else:
        if not args.preprocessed_root or not args.annotations_csv:
            raise ValueError("Either --source-csv or both --preprocessed-root and --annotations-csv are required.")
        build_holdout_csv(
            preprocessed_root=Path(args.preprocessed_root),
            annotations_csv=Path(args.annotations_csv),
            output_csv=Path(args.output_csv),
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
