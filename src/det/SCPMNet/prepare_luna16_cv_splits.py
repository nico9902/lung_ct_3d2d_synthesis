from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def find_series_dirs(preprocessed_root: Path) -> dict[str, tuple[int, Path, Path]]:
    series: dict[str, tuple[int, Path, Path]] = {}
    for fold in range(10):
        subset_dir = preprocessed_root / f"subset{fold}"
        if not subset_dir.exists():
            continue
        for series_dir in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
            seriesuid = series_dir.name
            volume_path = series_dir / f"{seriesuid}_volume.nii.gz"
            if not volume_path.exists():
                candidates = sorted(series_dir.glob("*_volume.nii.gz"))
                if not candidates:
                    continue
                volume_path = candidates[0]
            series[seriesuid] = (fold, series_dir, volume_path)
    return series


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def annotation_rows(
    seriesuid: str,
    split: str,
    image_path: Path,
    mask_path: Path | None,
    data_root: Path,
    annotations: pd.DataFrame,
) -> list[dict]:
    import SimpleITK as sitk

    rows: list[dict] = []
    image = sitk.ReadImage(str(image_path))
    nodules = annotations[annotations["seriesuid"].astype(str) == seriesuid]
    image_rel = relpath(image_path, data_root)
    mask_rel = relpath(mask_path, data_root) if mask_path is not None and mask_path.exists() else ""

    if nodules.empty:
        rows.append(
            {
                "seriesuid": seriesuid,
                "image_path": image_rel,
                "nodule_mask_path": mask_rel,
                "split": split,
                "x": None,
                "y": None,
                "z": None,
                "w": None,
                "h": None,
                "d": None,
                "diameter_mm": None,
                "label": "negative",
            }
        )
        return rows

    for _, nodule in nodules.iterrows():
        world_xyz = (
            float(nodule["coordX"]),
            float(nodule["coordY"]),
            float(nodule["coordZ"]),
        )
        x, y, z = image.TransformPhysicalPointToContinuousIndex(world_xyz)
        diameter = float(nodule["diameter_mm"])
        rows.append(
            {
                "seriesuid": seriesuid,
                "image_path": image_rel,
                "nodule_mask_path": mask_rel,
                "split": split,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "w": diameter,
                "h": diameter,
                "d": diameter,
                "diameter_mm": diameter,
                "label": "nodule",
            }
        )
    return rows


def build_fold_csvs(
    preprocessed_root: Path,
    annotations_csv: Path,
    output_dir: Path,
    val_offset: int,
) -> None:
    annotations = pd.read_csv(annotations_csv)
    series = find_series_dirs(preprocessed_root)
    if not series:
        raise ValueError(f"No subset*/<seriesuid> volume folders found under {preprocessed_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for test_fold in range(10):
        val_fold = (test_fold + val_offset) % 10
        rows = []
        for seriesuid, (subset_fold, series_dir, volume_path) in sorted(series.items()):
            if subset_fold == test_fold:
                split = "test"
            elif subset_fold == val_fold:
                split = "val"
            else:
                split = "train"
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

        fold_df = pd.DataFrame(rows)
        fold_path = output_dir / f"luna16_fold{test_fold}.csv"
        fold_df.to_csv(fold_path, index=False)

        split_stats = fold_df.groupby("split").agg(
            rows=("seriesuid", "size"),
            series=("seriesuid", "nunique"),
            nodules=("label", lambda s: int((s == "nodule").sum())),
        )
        for split, stats in split_stats.iterrows():
            summary_rows.append(
                {
                    "fold": test_fold,
                    "test_subset": f"subset{test_fold}",
                    "val_subset": f"subset{val_fold}",
                    "split": split,
                    "rows": int(stats["rows"]),
                    "series": int(stats["series"]),
                    "nodules": int(stats["nodules"]),
                    "csv_path": str(fold_path),
                }
            )
        print(f"wrote {fold_path}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(f"wrote {output_dir / 'summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SCPMNet CSV files for LUNA16 10-fold cross-validation.")
    parser.add_argument("--preprocessed-root", required=True, help="Root containing subset0 ... subset9 folders.")
    parser.add_argument("--annotations-csv", required=True, help="Official LUNA16 annotations.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory where luna16_fold*.csv files are written.")
    parser.add_argument(
        "--val-offset",
        type=int,
        default=1,
        help="Validation fold is (test_fold + val_offset) %% 10. Default: 1.",
    )
    args = parser.parse_args()

    build_fold_csvs(
        preprocessed_root=Path(args.preprocessed_root),
        annotations_csv=Path(args.annotations_csv),
        output_dir=Path(args.output_dir),
        val_offset=args.val_offset,
    )


if __name__ == "__main__":
    main()
