from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.det.SCPMNet.aggregate_luna16_cv import gt_from_rows


DEFAULT_TOPKS = tuple(range(1, 11))
DEFAULT_THRESHOLDS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)


def fold_index(path: Path) -> int:
    match = re.search(r"fold(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot infer fold index from {path}")
    return int(match.group(1))


def parse_numbers(values: Sequence[str], cast=float) -> list:
    parsed: list = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                parsed.append(cast(part))
    return parsed


def load_gt(split_path: Path) -> dict[str, np.ndarray]:
    split = pd.read_csv(split_path)
    test_rows = split[split["split"].astype(str) == "test"]
    return {str(seriesuid): gt_from_rows(rows) for seriesuid, rows in test_rows.groupby("seriesuid", sort=False)}


def match_selected(
    selected: pd.DataFrame,
    gt_by_series: dict[str, np.ndarray],
) -> tuple[dict[str, set[int]], int, int]:
    matched = {seriesuid: np.zeros(len(gt), dtype=bool) for seriesuid, gt in gt_by_series.items()}
    false_positives = 0
    true_positives = 0
    for _, row in selected.iterrows():
        seriesuid = str(row["seriesuid"])
        gt = gt_by_series.get(seriesuid, np.zeros((0, 4), dtype=np.float32))
        available = np.where(~matched.get(seriesuid, np.zeros(0, dtype=bool)))[0]
        match_idx = None
        if len(available):
            pred_center = row[["coordZ", "coordY", "coordX"]].to_numpy(dtype=np.float32)
            available_gt = gt[available]
            distances = np.linalg.norm(available_gt[:, :3] - pred_center.reshape(1, 3), axis=1)
            hits = distances <= available_gt[:, 3]
            if np.any(hits):
                hit_indices = np.where(hits)[0]
                best = hit_indices[int(np.argmin(distances[hit_indices]))]
                match_idx = int(available[best])
        if match_idx is None:
            false_positives += 1
        else:
            matched[seriesuid][match_idx] = True
            true_positives += 1
    covered = {seriesuid: set(np.where(flags)[0].astype(int).tolist()) for seriesuid, flags in matched.items()}
    return covered, true_positives, false_positives


def select_predictions(predictions: pd.DataFrame, score_col: str, threshold: float, topk: int) -> pd.DataFrame:
    eligible = predictions[predictions[score_col].astype(float) >= float(threshold)].copy()
    if eligible.empty:
        return eligible
    return (
        eligible.sort_values(["seriesuid", score_col], ascending=[True, False])
        .groupby("seriesuid", sort=False)
        .head(int(topk))
        .reset_index(drop=True)
    )


def positive_scan_count(gt_by_series: dict[str, np.ndarray]) -> int:
    return int(sum(1 for gt in gt_by_series.values() if len(gt)))


def summarize_selection(
    selected: pd.DataFrame,
    gt_by_series: dict[str, np.ndarray],
    fold: int | None,
    threshold: float,
    topk: int,
    scope: str,
) -> tuple[dict, dict[str, set[int]]]:
    covered, tp, fp = match_selected(selected, gt_by_series)
    total_nodules = int(sum(len(gt) for gt in gt_by_series.values()))
    total_scans = int(len(gt_by_series))
    positive_scans = positive_scan_count(gt_by_series)
    covered_positive_scans = int(sum(1 for seriesuid, gt in gt_by_series.items() if len(gt) and covered.get(seriesuid)))
    row = {
        "scope": scope,
        "fold": fold if fold is not None else "",
        "threshold": float(threshold),
        "topk": int(topk),
        "detections_kept": int(len(selected)),
        "true_positive_detections": int(tp),
        "false_positives": int(fp),
        "total_scans": total_scans,
        "false_positives_per_scan": float(fp / total_scans) if total_scans else 0.0,
        "covered_nodules": int(sum(len(v) for v in covered.values())),
        "total_nodules": total_nodules,
        "nodule_coverage": float(sum(len(v) for v in covered.values()) / total_nodules) if total_nodules else 0.0,
        "covered_positive_scans": covered_positive_scans,
        "positive_scans": positive_scans,
        "positive_scan_coverage": float(covered_positive_scans / positive_scans) if positive_scans else 0.0,
    }
    return row, covered


def missed_nodule_rows(gt_by_series: dict[str, np.ndarray], covered: dict[str, set[int]], fold: int, threshold: float, topk: int) -> list[dict]:
    rows: list[dict] = []
    for seriesuid, gt in gt_by_series.items():
        covered_indices = covered.get(seriesuid, set())
        for idx, (z, y, x, radius) in enumerate(gt.tolist()):
            if idx in covered_indices:
                continue
            rows.append(
                {
                    "fold": fold,
                    "threshold": float(threshold),
                    "topk": int(topk),
                    "seriesuid": seriesuid,
                    "gt_index": int(idx),
                    "coordZ": float(z),
                    "coordY": float(y),
                    "coordX": float(x),
                    "radius": float(radius),
                }
            )
    return rows


def plot_heatmap(pooled: pd.DataFrame, out_path: Path, value_col: str, title: str) -> None:
    pivot = pooled.pivot(index="threshold", columns="topk", values=value_col).sort_index(ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cell_w = 72
    cell_h = 34
    left = 94
    top = 66
    width = left + cell_w * len(pivot.columns) + 24
    height = top + cell_h * len(pivot.index) + 54

    def color(value: float) -> str:
        value = float(np.clip(value, 0.0, 1.0))
        stops = [
            (0.0, (68, 1, 84)),
            (0.25, (59, 82, 139)),
            (0.5, (33, 145, 140)),
            (0.75, (94, 201, 98)),
            (1.0, (253, 231, 37)),
        ]
        for (x0, c0), (x1, c1) in zip(stops[:-1], stops[1:]):
            if value <= x1:
                t = (value - x0) / (x1 - x0)
                rgb = tuple(int(round(c0[i] + t * (c1[i] - c0[i]))) for i in range(3))
                return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        return "#fde725"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700">{title}</text>',
        f'<text x="{width / 2:.1f}" y="{height - 10}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">Top-k per scan</text>',
        f'<text x="16" y="{top + cell_h * len(pivot.index) / 2:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" transform="rotate(-90 16 {top + cell_h * len(pivot.index) / 2:.1f})">Score threshold</text>',
    ]
    for j, topk in enumerate(pivot.columns):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="{top - 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">{int(topk)}</text>')
    for i, threshold in enumerate(pivot.index):
        y = top + i * cell_h + cell_h / 2 + 4
        parts.append(f'<text x="{left - 12}" y="{y:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11">{threshold:g}</text>')
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = float(pivot.iloc[i, j])
            x = left + j * cell_w
            y = top + i * cell_h
            fill = color(value)
            text_color = "white" if value < 0.58 else "black"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="white" stroke-width="1"/>')
            parts.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="11" fill="{text_color}">{100 * value:.1f}</text>'
            )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n")


def write_markdown(out_dir: Path, args: argparse.Namespace, pooled: pd.DataFrame, best_rows: pd.DataFrame) -> None:
    md = out_dir / "detection_coverage_report.md"
    best_nodule = best_rows.sort_values(["nodule_coverage", "false_positives_per_scan"], ascending=[False, True]).head(10)
    efficient = pooled[pooled["nodule_coverage"] >= 0.8].sort_values(["false_positives_per_scan", "topk", "threshold"]).head(10)
    table_cols = [
        "threshold",
        "topk",
        "covered_nodules",
        "total_nodules",
        "nodule_coverage",
        "positive_scan_coverage",
        "false_positives_per_scan",
        "detections_kept",
    ]

    def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        body = []
        for _, row in frame.iterrows():
            values = []
            for col in columns:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            body.append("| " + " | ".join(values) + " |")
        return "\n".join([header, sep, *body])

    lines = [
        "# LUNA16 Detection Coverage Map",
        "",
        f"Prediction root: `{args.prediction_root}`",
        f"Prediction file: `{args.prediction_name}`",
        f"Score column: `{args.score_col}`",
        "",
        "A GT nodule is covered when one selected detection center falls inside the GT nodule radius. Matching is one-to-one.",
        "",
        "![Nodule coverage heatmap](assets/nodule_coverage_heatmap.svg)",
        "",
        "## Best Pooled Coverage Rows",
        "",
        markdown_table(best_nodule[table_cols], table_cols),
        "",
        "## Efficient Rows With >=80% Nodule Coverage",
        "",
    ]
    if efficient.empty:
        lines.append("No pooled row reaches 80% nodule coverage.")
    else:
        lines.append(markdown_table(efficient[table_cols], table_cols))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `coverage_by_threshold_topk.csv`: fold and pooled coverage for every threshold/top-k pair.",
            "- `pooled_coverage.csv`: pooled rows only.",
            "- `missed_nodules.csv`: missed GT nodules for every threshold/top-k pair.",
            "- `assets/nodule_coverage_heatmap.svg`: pooled nodule coverage map.",
            "- `assets/positive_scan_coverage_heatmap.svg`: pooled positive-scan coverage map.",
        ]
    )
    md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LUNA16 GT nodule coverage for detection top-k/threshold working points.")
    parser.add_argument("--prediction-root", type=Path, default=Path("outputs/scpmnet_luna16_10fold"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/LUNA16_preprocessed/cv_splits"))
    parser.add_argument("--fold-glob", default="scpmnet_paper_luna16_fold*")
    parser.add_argument("--prediction-name", default="test_predictions.csv")
    parser.add_argument("--score-col", default="probability")
    parser.add_argument("--topks", nargs="+", default=[str(v) for v in DEFAULT_TOPKS])
    parser.add_argument("--thresholds", nargs="+", default=[str(v) for v in DEFAULT_THRESHOLDS])
    parser.add_argument("--out-dir", type=Path, default=Path("docs/luna16_detection_coverage_map"))
    parser.add_argument("--write-missed-for-all", action="store_true")
    args = parser.parse_args()

    topks = sorted(set(parse_numbers(args.topks, int)))
    thresholds = sorted(set(parse_numbers(args.thresholds, float)))
    fold_dirs = sorted(args.prediction_root.glob(args.fold_glob), key=fold_index)
    if len(fold_dirs) != 10:
        raise ValueError(f"Expected 10 fold directories, found {len(fold_dirs)} in {args.prediction_root}")

    all_predictions = []
    all_gt: dict[str, np.ndarray] = {}
    fold_data = []
    for fold_dir in fold_dirs:
        fold = fold_index(fold_dir)
        pred_path = fold_dir / "predictions" / args.prediction_name
        split_path = args.split_dir / f"luna16_fold{fold}.csv"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        predictions = pd.read_csv(pred_path)
        if args.score_col not in predictions.columns:
            raise ValueError(f"{pred_path} does not contain score column {args.score_col!r}")
        gt = load_gt(split_path)
        fold_data.append((fold, predictions, gt))
        all_predictions.append(predictions.assign(fold=fold))
        all_gt.update(gt)

    rows = []
    missed_rows = []
    pooled_predictions = pd.concat(all_predictions, ignore_index=True)
    best_pooled_key = None
    best_pooled_coverage = -1.0
    pooled_covered_by_key: dict[tuple[float, int], dict[str, set[int]]] = {}

    for threshold in thresholds:
        for topk in topks:
            for fold, predictions, gt in fold_data:
                selected = select_predictions(predictions, args.score_col, threshold, topk)
                row, covered = summarize_selection(selected, gt, fold, threshold, topk, f"fold_{fold}")
                rows.append(row)
                if args.write_missed_for_all:
                    missed_rows.extend(missed_nodule_rows(gt, covered, fold, threshold, topk))

            pooled_selected = select_predictions(pooled_predictions, args.score_col, threshold, topk)
            pooled_row, pooled_covered = summarize_selection(pooled_selected, all_gt, None, threshold, topk, "pooled")
            rows.append(pooled_row)
            key = (float(threshold), int(topk))
            pooled_covered_by_key[key] = pooled_covered
            if pooled_row["nodule_coverage"] > best_pooled_coverage:
                best_pooled_coverage = float(pooled_row["nodule_coverage"])
                best_pooled_key = key

    if not args.write_missed_for_all and best_pooled_key is not None:
        threshold, topk = best_pooled_key
        for fold, _, gt in fold_data:
            fold_covered = {seriesuid: covered for seriesuid, covered in pooled_covered_by_key[best_pooled_key].items() if seriesuid in gt}
            missed_rows.extend(missed_nodule_rows(gt, fold_covered, fold, threshold, topk))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = args.out_dir / "assets"
    coverage = pd.DataFrame(rows)
    coverage.to_csv(args.out_dir / "coverage_by_threshold_topk.csv", index=False)
    pooled = coverage[coverage["scope"] == "pooled"].copy().sort_values(["threshold", "topk"])
    pooled.to_csv(args.out_dir / "pooled_coverage.csv", index=False)
    pd.DataFrame(missed_rows).to_csv(args.out_dir / "missed_nodules.csv", index=False)

    plot_heatmap(pooled, assets_dir / "nodule_coverage_heatmap.svg", "nodule_coverage", "Pooled GT Nodule Coverage")
    plot_heatmap(pooled, assets_dir / "positive_scan_coverage_heatmap.svg", "positive_scan_coverage", "Pooled Positive-Scan Coverage")
    write_markdown(args.out_dir, args, pooled, pooled)

    best = pooled.sort_values(["nodule_coverage", "false_positives_per_scan"], ascending=[False, True]).iloc[0]
    print(f"Wrote coverage analysis to {args.out_dir}")
    print(
        "Best pooled nodule coverage: "
        f"threshold={best.threshold:g} topk={int(best.topk)} "
        f"coverage={100 * best.nodule_coverage:.2f}% "
        f"positive_scan_coverage={100 * best.positive_scan_coverage:.2f}% "
        f"FP/scan={best.false_positives_per_scan:.3f}"
    )


if __name__ == "__main__":
    main()
