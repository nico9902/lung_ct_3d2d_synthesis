from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


METRIC_COLUMN_RES = [
    re.compile(r"^backbone:\s*(?P<backbone>.+?)\s*-\s*(?P<metric>test_(?!.*__).+?)$"),
    re.compile(r"^fold_(?P<fold>\d+)_(?P<backbone>.+?)\s*-\s*(?P<metric>test_(?!.*__).+?)$"),
]


def parse_metric_exports(output_dir: Path) -> pd.DataFrame:
    fold_metrics = parse_fold_metric_files(output_dir)
    if not fold_metrics.empty:
        return fold_metrics

    rows: list[dict[str, object]] = []
    for csv_path in sorted(output_dir.rglob("wandb_export*.csv")):
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue

        experiment = _relative_parent(csv_path, output_dir)
        for column in frame.columns:
            metric_info = _parse_metric_column(column)
            if metric_info is None:
                continue

            fold, backbone, metric = metric_info
            values = pd.to_numeric(frame[column], errors="coerce")
            source_min_values = _numeric_column_or_none(frame, f"{column}__MIN")
            source_max_values = _numeric_column_or_none(frame, f"{column}__MAX")
            source_std_values = _numeric_column_or_none(frame, f"{column}__STD")
            for row_index, value in values.dropna().items():
                rows.append(
                    {
                        "experiment": experiment,
                        "fold": fold,
                        "row": int(row_index),
                        "backbone": backbone,
                        "metric": metric,
                        "value": float(value),
                        "source_min": _series_value_or_none(source_min_values, row_index),
                        "source_max": _series_value_or_none(source_max_values, row_index),
                        "source_std": _series_value_or_none(source_std_values, row_index),
                        "source_csv": str(csv_path),
                    }
                )

    if not rows:
        raise RuntimeError(f"No W&B test metric columns found under {output_dir}")
    raw = pd.DataFrame(rows)
    if raw["fold"].notna().any():
        has_fold_values = raw.loc[raw["fold"].notna(), ["experiment", "backbone", "metric"]].drop_duplicates()
        raw = raw.merge(
            has_fold_values.assign(_has_fold_values=True),
            on=["experiment", "backbone", "metric"],
            how="left",
        )
        raw = raw[raw["fold"].notna() | raw["_has_fold_values"].isna()].drop(columns="_has_fold_values").copy()
    raw["metric_min"] = raw["source_min"].combine_first(raw["value"])
    raw["metric_max"] = raw["source_max"].combine_first(raw["value"])
    return raw


def _parse_metric_column(column: str) -> tuple[int | None, str, str] | None:
    for pattern in METRIC_COLUMN_RES:
        match = pattern.match(column)
        if match is None:
            continue
        fold = match.groupdict().get("fold")
        return (
            int(fold) if fold is not None else None,
            match.group("backbone").strip(),
            match.group("metric").strip(),
        )
    return None


def parse_fold_metric_files(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(output_dir.glob("fold_*/*/test_metrics.json")):
        fold_name = metrics_path.parent.parent.name
        backbone = metrics_path.parent.name
        try:
            fold = int(fold_name.split("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            fold = None

        with metrics_path.open() as handle:
            metrics = json.load(handle)

        for metric, value in metrics.items():
            if not metric.startswith("test_"):
                continue
            value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "experiment": output_dir.name,
                    "fold": fold,
                    "backbone": backbone,
                    "metric": metric,
                    "value": float(value),
                    "source_min": None,
                    "source_max": None,
                    "source_std": None,
                    "source_csv": str(metrics_path),
                }
            )

    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    raw["metric_min"] = raw["value"]
    raw["metric_max"] = raw["value"]
    return raw


def _relative_parent(csv_path: Path, output_dir: Path) -> str:
    parent = csv_path.parent
    if parent == output_dir:
        return output_dir.name
    return str(parent.relative_to(output_dir))


def _numeric_or_none(value) -> float | None:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _numeric_column_or_none(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame.columns:
        return None
    return pd.to_numeric(frame[column], errors="coerce")


def _series_value_or_none(values: pd.Series | None, row_index: int) -> float | None:
    if values is None:
        return None
    value = values.get(row_index)
    if pd.isna(value):
        return None
    return float(value)


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        raw.groupby(["experiment", "backbone", "metric"], dropna=False)
        .agg(
            count=("value", "count"),
            mean=("value", "mean"),
            value_std=("value", _std_or_none),
            exported_std=("source_std", _mean_or_none),
            min=("metric_min", "min"),
            max=("metric_max", "max"),
        )
        .reset_index()
    )
    grouped["std"] = grouped["value_std"]
    missing_std = grouped["std"].isna()
    if missing_std.any():
        grouped.loc[missing_std, "std"] = grouped.loc[missing_std, "exported_std"]
    grouped = grouped.drop(columns=["value_std", "exported_std"])
    summary = (
        grouped.set_index(["experiment", "backbone", "metric"])[["count", "mean", "std", "min", "max"]]
        .unstack("metric")
        .sort_index()
    )
    summary.columns = [f"{metric}_{stat}" for stat, metric in summary.columns]
    summary = summary.reset_index()

    metric_order = sorted(raw["metric"].unique())
    ordered_columns = ["experiment", "backbone"]
    for metric in metric_order:
        ordered_columns.extend(
            [
                f"{metric}_count",
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_min",
                f"{metric}_max",
            ]
        )
    return summary[[column for column in ordered_columns if column in summary.columns]]


def _std_or_none(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return None
    return float(values.std(ddof=1))


def _mean_or_none(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def write_workbook(raw: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        raw.sort_values(["experiment", "backbone", "metric"]).to_excel(writer, sheet_name="raw_metrics", index=False)

        for sheet_name in ("summary", "raw_metrics"):
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                width = min(max(max_length + 2, 12), 80)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an Excel summary with mean/std test metrics for each synthetic 2D backbone."
    )
    parser.add_argument("--output-dir", default="outputs/luna16_synthetic_2d")
    parser.add_argument("--xlsx-name", default="backbone_summary.xlsx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw = parse_metric_exports(output_dir)
    summary = build_summary(raw)
    output_path = output_dir / args.xlsx_name
    write_workbook(raw, summary, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
