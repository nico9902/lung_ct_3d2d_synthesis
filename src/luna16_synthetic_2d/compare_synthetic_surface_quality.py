"""Compare QC metrics for two LUNA16 synthetic surface output roots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


SUMMARY_METRICS = [
    "foreground_fraction",
    "foreground_std",
    "z_range",
    "z_std",
    "grad_p99",
    "grad_max",
    "qc_badness",
]

QC_BUCKETS = [
    ("good_or_low_review", -np.inf, 0.10),
    ("moderate_review", 0.10, 0.30),
    ("high_review", 0.30, np.inf),
]


def find_one(sample_dir: Path, prefix: str, suffix: str) -> Path | None:
    matches = sorted(sample_dir.glob(f"{prefix}*{suffix}"))
    return matches[0] if matches else None


def image_metrics(path: Path) -> dict[str, float]:
    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    foreground = arr > 0.0
    if not np.any(foreground):
        return {
            "foreground_fraction": 0.0,
            "foreground_mean": 0.0,
            "foreground_std": 0.0,
            "image_p99": 0.0,
        }

    values = arr[foreground]
    return {
        "foreground_fraction": float(foreground.mean()),
        "foreground_mean": float(values.mean()),
        "foreground_std": float(values.std()),
        "image_p99": float(np.percentile(values, 99)),
    }


def surface_metrics(path: Path) -> dict[str, float]:
    grid = np.load(path).astype(np.float32)
    gy, gx = np.gradient(grid)
    grad = np.sqrt(gx * gx + gy * gy)
    edge = np.concatenate([grid[0], grid[-1], grid[:, 0], grid[:, -1]])
    return {
        "z_range": float(np.nanmax(grid) - np.nanmin(grid)),
        "z_std": float(np.nanstd(grid)),
        "grad_p95": float(np.nanpercentile(grad, 95)),
        "grad_p99": float(np.nanpercentile(grad, 99)),
        "grad_max": float(np.nanmax(grad)),
        "edge_z_std": float(np.nanstd(edge)),
    }


def control_metrics(sample_dir: Path, seriesuid: str) -> dict[str, float | int]:
    control_path = sample_dir / f"control_points_{seriesuid}.npy"
    label_path = sample_dir / f"point_labels_{seriesuid}.npy"
    if not control_path.exists():
        return {
            "control_points": 0,
            "control_labels": 0,
            "anchor_labels": 0,
            "control_z_range": 0.0,
        }

    points = np.load(control_path)
    labels = np.load(label_path, allow_pickle=True) if label_path.exists() else np.array([])
    label_text = np.asarray(labels).astype(str) if labels.size else np.array([])
    control_mask = np.char.find(label_text, "control") >= 0 if label_text.size else np.ones(len(points), dtype=bool)
    anchor_mask = np.char.find(label_text, "anchor") >= 0 if label_text.size else np.zeros(len(points), dtype=bool)
    control_points = points[control_mask] if len(control_mask) == len(points) else points
    control_z_range = float(np.nanmax(control_points[:, 0]) - np.nanmin(control_points[:, 0])) if len(control_points) else 0.0
    return {
        "control_points": int(len(points)),
        "control_labels": int(control_mask.sum()) if len(control_mask) == len(points) else int(len(points)),
        "anchor_labels": int(anchor_mask.sum()) if len(anchor_mask) == len(points) else 0,
        "control_z_range": control_z_range,
    }


def qc_badness(row: dict[str, float]) -> float:
    """Small heuristic score for triage; lower is better."""
    low_foreground = max(0.0, (0.35 - row["foreground_fraction"]) / 0.35)
    low_contrast = max(0.0, (0.20 - row["foreground_std"]) / 0.20)
    roughness = row["grad_p99"] / 10.0
    z_span = row["z_range"] / 180.0
    edge_variation = row["edge_z_std"] / 45.0
    raw = (
        0.30 * min(z_span, 2.0)
        + 0.30 * min(roughness, 2.0)
        + 0.20 * min(low_foreground, 2.0)
        + 0.10 * min(low_contrast, 2.0)
        + 0.10 * min(edge_variation, 2.0)
    )
    return float(min(raw, 1.0))


def qc_bucket(score: float) -> str:
    for label, low, high in QC_BUCKETS:
        if low < score <= high:
            return label
    return QC_BUCKETS[-1][0]


def scan_root(root: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        seriesuid = sample_dir.name
        png = find_one(sample_dir, "surface_", ".png")
        grid = find_one(sample_dir, "surface_grid_float_", ".npy")
        if png is None or grid is None:
            continue

        row: dict[str, object] = {
            "source": label,
            "seriesuid": seriesuid,
            "png": str(png),
            "grid": str(grid),
        }
        row.update(image_metrics(png))
        row.update(surface_metrics(grid))
        row.update(control_metrics(sample_dir, seriesuid))
        row["qc_badness"] = qc_badness(row)  # type: ignore[arg-type]
        row["qc_bucket"] = qc_bucket(float(row["qc_badness"]))
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("qc_badness", ascending=False).reset_index(drop=True)
        df["qc_rank_worst"] = np.arange(1, len(df) + 1)
    return df


def describe(df: pd.DataFrame) -> pd.DataFrame:
    return df[SUMMARY_METRICS].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])


def paired_comparison(gt: pd.DataFrame, detector: pd.DataFrame) -> pd.DataFrame:
    gt_cols = ["seriesuid", *SUMMARY_METRICS, "control_labels", "anchor_labels", "control_z_range"]
    det_cols = ["seriesuid", *SUMMARY_METRICS, "control_labels", "anchor_labels", "control_z_range"]
    paired = gt[gt_cols].merge(detector[det_cols], on="seriesuid", suffixes=("_gt", "_detector"))
    for metric in SUMMARY_METRICS:
        paired[f"{metric}_delta_detector_minus_gt"] = paired[f"{metric}_detector"] - paired[f"{metric}_gt"]
    paired["detector_better_qc"] = paired["qc_badness_delta_detector_minus_gt"] < 0
    paired = paired.sort_values("qc_badness_delta_detector_minus_gt", ascending=False).reset_index(drop=True)
    return paired


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    view = df.loc[:, columns]
    if limit is not None:
        view = view.head(limit)
    return view.to_markdown(index=False, floatfmt=".3f")


def write_report(out_dir: Path, gt: pd.DataFrame, detector: pd.DataFrame, paired: pd.DataFrame) -> None:
    summary_rows = []
    for name, df in [("ground_truth", gt), ("detector_top5", detector)]:
        row = {"source": name, "n": int(len(df))}
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = float(df[metric].mean())
            row[f"{metric}_median"] = float(df[metric].median())
            row[f"{metric}_p95"] = float(df[metric].quantile(0.95))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    delta_cols = [c for c in paired.columns if c.endswith("_delta_detector_minus_gt")]
    delta_summary = paired[delta_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    delta_summary.index.name = "metric"
    delta_summary = delta_summary.reset_index()

    better = int(paired["detector_better_qc"].sum())
    worse = int((~paired["detector_better_qc"]).sum())
    paired_n = int(len(paired))
    bucket_summary = []
    for name, df in [("ground_truth", gt), ("detector_top5", detector)]:
        counts = df["qc_bucket"].value_counts()
        for label, _, _ in QC_BUCKETS:
            bucket_summary.append({"source": name, "qc_bucket": label, "n": int(counts.get(label, 0))})
    bucket_summary_df = pd.DataFrame(bucket_summary)

    control_summary = []
    for name, df in [("ground_truth", gt), ("detector_top5", detector)]:
        grouped = df.groupby("control_labels", sort=True)["qc_badness"].agg(["count", "mean", "median"]).reset_index()
        grouped.insert(0, "source", name)
        control_summary.append(grouped)
    control_summary_df = pd.concat(control_summary, ignore_index=True)

    report = [
        "# LUNA16 Detector vs Ground-Truth Synthetic Surface QC",
        "",
        "Lower `qc_badness` is better. The score is a triage heuristic combining foreground coverage, foreground contrast, z-surface range, edge variation, and surface gradient roughness.",
        "",
        f"- Ground-truth generated cases analyzed: **{len(gt)}**",
        f"- Detector top-5 generated cases analyzed: **{len(detector)}**",
        f"- Paired cases compared by `seriesuid`: **{paired_n}**",
        f"- Detector lower/better `qc_badness`: **{better} / {paired_n} ({better / paired_n:.1%})**",
        f"- Detector equal-or-higher/worse `qc_badness`: **{worse} / {paired_n} ({worse / paired_n:.1%})**",
        "",
        "## Source Summary",
        "",
        markdown_table(
            summary,
            [
                "source",
                "n",
                "qc_badness_mean",
                "qc_badness_median",
                "qc_badness_p95",
                "foreground_fraction_mean",
                "z_range_mean",
                "grad_p99_mean",
            ],
        ),
        "",
        "## QC Buckets",
        "",
        "`good_or_low_review` is `qc_badness <= 0.10`, `moderate_review` is `0.10 < qc_badness <= 0.30`, and `high_review` is `qc_badness > 0.30`.",
        "",
        markdown_table(bucket_summary_df, ["source", "qc_bucket", "n"]),
        "",
        "## Likely Explanation",
        "",
        "The detector-top5 run uses five detector candidates for every scan. In these saved outputs, each nodule/detection region contributes five labelled control points, so detector-top5 has `25` control labels for every case. Ground-truth generation is much less constrained: most cases have one or two nodule regions, corresponding to `5` or `10` control labels.",
        "",
        "This matches the QC failure mode: more control regions force the interpolated surface to satisfy points across a larger cranio-caudal spread, increasing `z_range` and `grad_p99`. The foreground image statistics stay similar, but the detector surfaces are geometrically rougher.",
        "",
        markdown_table(control_summary_df, ["source", "control_labels", "count", "mean", "median"]),
        "",
        "## Paired Delta Summary",
        "",
        "Delta is detector minus ground truth; negative `qc_badness` means detector is better by this heuristic.",
        "",
        markdown_table(delta_summary, ["metric", "mean", "std", "5%", "25%", "50%", "75%", "95%"]),
        "",
        "## Detector Most Worse Than Ground Truth",
        "",
        markdown_table(
            paired,
            [
                "seriesuid",
                "qc_badness_gt",
                "qc_badness_detector",
                "qc_badness_delta_detector_minus_gt",
                "z_range_gt",
                "z_range_detector",
                "grad_p99_gt",
                "grad_p99_detector",
                "foreground_fraction_gt",
                "foreground_fraction_detector",
            ],
            limit=12,
        ),
        "",
        "## Detector Most Better Than Ground Truth",
        "",
        markdown_table(
            paired.sort_values("qc_badness_delta_detector_minus_gt", ascending=True),
            [
                "seriesuid",
                "qc_badness_gt",
                "qc_badness_detector",
                "qc_badness_delta_detector_minus_gt",
                "z_range_gt",
                "z_range_detector",
                "grad_p99_gt",
                "grad_p99_detector",
                "foreground_fraction_gt",
                "foreground_fraction_detector",
            ],
            limit=12,
        ),
        "",
        "## Output Files",
        "",
        "- `ground_truth_surface_qc_metrics.csv`",
        "- `detector_top5_surface_qc_metrics.csv`",
        "- `paired_surface_qc_comparison.csv`",
    ]
    (out_dir / "detector_vs_ground_truth_surface_qc_report.md").write_text("\n".join(report) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, default=Path("outputs/luna16_saliency_synthetic_gt"))
    parser.add_argument("--detector-root", type=Path, default=Path("outputs/luna16_saliency_synthetic_detector_top5"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/luna16_saliency_synthetic_detector_top5_qc_report"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt = scan_root(args.gt_root, "ground_truth")
    detector = scan_root(args.detector_root, "detector_top5")
    paired = paired_comparison(gt, detector)

    gt.to_csv(args.out_dir / "ground_truth_surface_qc_metrics.csv", index=False)
    detector.to_csv(args.out_dir / "detector_top5_surface_qc_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "paired_surface_qc_comparison.csv", index=False)
    describe(gt).to_csv(args.out_dir / "ground_truth_summary.csv")
    describe(detector).to_csv(args.out_dir / "detector_top5_summary.csv")
    write_report(args.out_dir, gt, detector, paired)

    print(f"Wrote {len(gt)} GT metrics, {len(detector)} detector metrics, {len(paired)} paired comparisons to {args.out_dir}")


if __name__ == "__main__":
    main()
