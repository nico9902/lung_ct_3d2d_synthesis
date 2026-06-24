from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROCESSING_RE = re.compile(r"Processing patient:\s*(?P<seriesuid>\S+)")
FOUND_RE = re.compile(r"Found\s+(?P<count>\d+)\s+nodules\s+for\s+patient\s+(?P<seriesuid>\S+)")
SERIESUID_RE = re.compile(r"1\.3\.6\.1\.4\.1\.14519\.5\.2\.1\.6279\.6001\.\d+")


def parse_positive_scans_from_log(log_path: Path) -> dict[str, int]:
    positive_scans: dict[str, int] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        match = FOUND_RE.search(line)
        if not match:
            continue
        seriesuid = match.group("seriesuid")
        count = int(match.group("count"))
        if count > 0:
            positive_scans[seriesuid] = max(count, positive_scans.get(seriesuid, 0))
    return positive_scans


def parse_scans_from_output_root(output_root: Path) -> dict[str, int]:
    seriesuids: set[str] = set()
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    for path in output_root.rglob("*"):
        matches = SERIESUID_RE.findall(str(path))
        seriesuids.update(matches)

    return {seriesuid: 0 for seriesuid in sorted(seriesuids)}


def load_output_control_points(sample_dir: Path, seriesuid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    control_path = sample_dir / f"control_points_{seriesuid}.npy"
    labels_path = sample_dir / f"point_labels_{seriesuid}.npy"
    grid_path = sample_dir / f"surface_grid_int_{seriesuid}.npy"
    if not grid_path.exists():
        grid_path = sample_dir / f"surface_grid_{seriesuid}.npy"
    if not control_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Missing control point files for {seriesuid} in {sample_dir}")

    control_points = np.load(control_path).astype(np.float32)
    point_labels = np.load(labels_path, allow_pickle=True).astype(str)
    surface_grid = np.load(grid_path).astype(np.float32) if grid_path.exists() else None
    return control_points, point_labels, surface_grid


def build_distribution_from_output_control_points(
    output_root: Path,
    output_dir: Path,
    positive_scans: dict[str, int] | None = None,
) -> tuple[Path, Path, Path]:
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    point_rows = []
    nodule_rows = []
    counts = []
    skipped = []
    skipped_not_positive = []
    for sample_dir in sorted(p for p in output_root.iterdir() if p.is_dir() and p.name != "hydra"):
        seriesuid = sample_dir.name
        if positive_scans is not None and seriesuid not in positive_scans:
            skipped_not_positive.append(seriesuid)
            continue

        try:
            control_points, point_labels, surface_grid = load_output_control_points(sample_dir, seriesuid)
        except FileNotFoundError:
            skipped.append(seriesuid)
            continue

        control_mask = point_labels == "control"
        controls = control_points[control_mask]
        if len(controls) == 0:
            skipped.append(seriesuid)
            continue

        if surface_grid is not None:
            h, w = surface_grid.shape
            max_z = max(float(np.max(surface_grid)), float(np.max(controls[:, 0])))
        else:
            h = int(np.ceil(np.max(control_points[:, 1]))) + 1
            w = int(np.ceil(np.max(control_points[:, 2]))) + 1
            max_z = float(np.max(controls[:, 0]))
        depth = max(max_z + 1.0, 1.0)

        nodule_count = int(len(controls) // 5)
        if len(controls) % 5 != 0:
            print(f"Warning: {seriesuid} has {len(controls)} control points, not divisible by 5.")

        counts.append({
            "seriesuid": seriesuid,
            "control_point_count": int(len(controls)),
            "nodule_count": nodule_count,
            "log_nodule_count": int(positive_scans.get(seriesuid, 0)) if positive_scans else 0,
        })
        for point_index, (z, y, x) in enumerate(controls):
            point_rows.append(
                {
                    "seriesuid": seriesuid,
                    "point_index": point_index,
                    "nodule_index": int(point_index // 5),
                    "z": float(z),
                    "y": float(y),
                    "x": float(x),
                    "rel_z": float(np.clip(z / max(depth - 1.0, 1.0), 0.0, 1.0)),
                    "rel_y": float(np.clip(y / max(h - 1.0, 1.0), 0.0, 1.0)),
                    "rel_x": float(np.clip(x / max(w - 1.0, 1.0), 0.0, 1.0)),
                    "grid_h": int(h),
                    "grid_w": int(w),
                    "estimated_depth": float(depth),
                }
            )

        for nodule_index in range(nodule_count):
            nodule_points = controls[nodule_index * 5 : (nodule_index + 1) * 5]
            z, y, x = nodule_points[0]
            nodule_rows.append(
                {
                    "seriesuid": seriesuid,
                    "nodule_index": nodule_index,
                    "z": float(z),
                    "y": float(y),
                    "x": float(x),
                    "rel_z": float(np.clip(z / max(depth - 1.0, 1.0), 0.0, 1.0)),
                    "rel_y": float(np.clip(y / max(h - 1.0, 1.0), 0.0, 1.0)),
                    "rel_x": float(np.clip(x / max(w - 1.0, 1.0), 0.0, 1.0)),
                    "grid_h": int(h),
                    "grid_w": int(w),
                    "estimated_depth": float(depth),
                    "control_point_count": int(len(nodule_points)),
                }
            )

    positions_df = pd.DataFrame(nodule_rows)
    point_positions_df = pd.DataFrame(point_rows)
    count_df = pd.DataFrame(counts)
    if positions_df.empty:
        raise RuntimeError(f"No control points found in {output_root}.")

    count_distribution_df = (
        count_df.groupby("nodule_count")
        .size()
        .reset_index(name="scan_count")
        .sort_values("nodule_count")
    )
    count_distribution_df["probability"] = (
        count_distribution_df["scan_count"] / count_distribution_df["scan_count"].sum()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    positions_csv = output_dir / "empirical_nodule_positions_from_control_points.csv"
    point_positions_csv = output_dir / "empirical_control_point_positions.csv"
    counts_csv = output_dir / "empirical_nodule_count_distribution_from_control_points.csv"
    npz_path = output_dir / "empirical_nodule_distribution_from_control_points.npz"
    json_path = output_dir / "empirical_nodule_distribution_from_control_points_summary.json"
    md_path = output_dir / "empirical_nodule_distribution_from_control_points_report.md"

    positions_df.to_csv(positions_csv, index=False)
    point_positions_df.to_csv(point_positions_csv, index=False)
    count_distribution_df.to_csv(counts_csv, index=False)
    np.savez_compressed(
        npz_path,
        relative_zyx=positions_df[["rel_z", "rel_y", "rel_x"]].to_numpy(dtype=np.float32),
        nodule_counts=count_df["nodule_count"].to_numpy(dtype=np.int16),
        seriesuids=count_df["seriesuid"].to_numpy(dtype=str),
        count_values=count_distribution_df["nodule_count"].to_numpy(dtype=np.int16),
        count_probabilities=count_distribution_df["probability"].to_numpy(dtype=np.float32),
    )

    summary = {
        "source": "control_points_grouped_as_nodules",
        "output_root": str(output_root),
        "positive_log_filter": positive_scans is not None,
        "positive_scans_in_log": len(positive_scans) if positive_scans is not None else None,
        "samples_with_control_points": int(count_df.shape[0]),
        "control_points": int(point_positions_df.shape[0]),
        "nodules": int(positions_df.shape[0]),
        "skipped_samples": skipped,
        "skipped_not_positive_in_log": skipped_not_positive,
        "relative_position_mean_zyx": positions_df[["rel_z", "rel_y", "rel_x"]].mean().round(6).to_dict(),
        "relative_position_std_zyx": positions_df[["rel_z", "rel_y", "rel_x"]].std().round(6).to_dict(),
        "relative_position_quantiles_zyx": {
            q: positions_df[["rel_z", "rel_y", "rel_x"]].quantile(float(q)).round(6).to_dict()
            for q in ["0.05", "0.25", "0.50", "0.75", "0.95"]
        },
        "nodule_count_distribution": {
            str(int(row["nodule_count"])): {
                "scan_count": int(row["scan_count"]),
                "probability": float(row["probability"]),
            }
            for _, row in count_distribution_df.iterrows()
        },
    }
    json_path.write_text(json.dumps(summary, indent=2))

    md = [
        "# Empirical LUNA16 Nodule Position Distribution From Control Points",
        "",
        f"Source output root: `{output_root}`",
        "",
        "This distribution is derived only from `control_points_*.npy`, `point_labels_*.npy`, and surface-grid files saved in the output folders.",
        "Every 5 control points are treated as one nodule; the first point in each group is the nodule center used for the empirical position distribution.",
        "It does **not** use `LUNA16_preprocessed` metadata or annotation CSV coordinates.",
        "",
        "## Summary",
        "",
        f"- Samples with control points: **{summary['samples_with_control_points']}**",
        f"- Control points: **{summary['control_points']}**",
        f"- Nodules: **{summary['nodules']}**",
        f"- Positive-log filter enabled: **{summary['positive_log_filter']}**",
        f"- Positive scans in log: **{summary['positive_scans_in_log']}**",
        f"- Skipped because not positive in log: **{len(skipped_not_positive)}**",
        f"- Skipped samples: **{len(skipped)}**",
        "",
        "## Relative Position Distribution",
        "",
        "Nodule-center coordinates are normalized per saved surface grid and saved in `[rel_z, rel_y, rel_x]`.",
        "",
        "| Axis | Mean | Std | Q05 | Q25 | Q50 | Q75 | Q95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for axis in ["rel_z", "rel_y", "rel_x"]:
        md.append(
            "| {axis} | {mean:.3f} | {std:.3f} | {q05:.3f} | {q25:.3f} | {q50:.3f} | {q75:.3f} | {q95:.3f} |".format(
                axis=axis,
                mean=summary["relative_position_mean_zyx"][axis],
                std=summary["relative_position_std_zyx"][axis],
                q05=summary["relative_position_quantiles_zyx"]["0.05"][axis],
                q25=summary["relative_position_quantiles_zyx"]["0.25"][axis],
                q50=summary["relative_position_quantiles_zyx"]["0.50"][axis],
                q75=summary["relative_position_quantiles_zyx"]["0.75"][axis],
                q95=summary["relative_position_quantiles_zyx"]["0.95"][axis],
            )
        )
    md.extend(
        [
            "",
            "## Nodule Count Distribution Per Scan",
            "",
            "| Nodules per scan | Scan count | Probability |",
            "|---:|---:|---:|",
        ]
    )
    for _, row in count_distribution_df.iterrows():
        md.append(f"| {int(row['nodule_count'])} | {int(row['scan_count'])} | {float(row['probability']):.4f} |")
    md.extend(
        [
            "",
            "## Files",
            "",
            f"- Nodule positions CSV: `{positions_csv}`",
            f"- Control point positions CSV: `{point_positions_csv}`",
            f"- Count distribution CSV: `{counts_csv}`",
            f"- Sampling arrays NPZ: `{npz_path}`",
            f"- JSON summary: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(md))
    return positions_csv, counts_csv, md_path


def load_metadata(preprocessed_root: Path, seriesuid: str) -> dict:
    metadata_path = preprocessed_root / seriesuid / f"{seriesuid}_metadata.json"
    with metadata_path.open() as f:
        return json.load(f)


def normalized_position(row: pd.Series, size_xyz: list[float]) -> tuple[float, float, float]:
    size_x, size_y, size_z = [max(float(v) - 1.0, 1.0) for v in size_xyz]
    rel_x = float(row["x"]) / size_x
    rel_y = float(row["y"]) / size_y
    rel_z = float(row["z"]) / size_z
    return (
        float(np.clip(rel_z, 0.0, 1.0)),
        float(np.clip(rel_y, 0.0, 1.0)),
        float(np.clip(rel_x, 0.0, 1.0)),
    )


def build_distribution(
    log_path: Path,
    output_root: Path,
    source: str,
    labels_csv: Path,
    preprocessed_root: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    if source == "log":
        positive_scans = parse_positive_scans_from_log(log_path)
        source_description = str(log_path)
    elif source == "output":
        positive_scans = parse_scans_from_output_root(output_root)
        source_description = str(output_root)
    elif source == "auto":
        if output_root.exists():
            positive_scans = parse_scans_from_output_root(output_root)
            source_description = str(output_root)
        else:
            positive_scans = parse_positive_scans_from_log(log_path)
            source_description = str(log_path)
    else:
        raise ValueError(f"Unsupported source: {source}")

    labels = pd.read_csv(labels_csv)
    labels = labels[labels["seriesuid"].astype(str).isin(positive_scans)].copy()
    labels = labels[labels["label"].astype(str).str.lower().eq("nodule")].copy()

    nodule_rows = []
    missing_metadata = []
    for _, row in labels.iterrows():
        seriesuid = str(row["seriesuid"])
        try:
            metadata = load_metadata(preprocessed_root, seriesuid)
        except FileNotFoundError:
            missing_metadata.append(seriesuid)
            continue

        rel_z, rel_y, rel_x = normalized_position(row, metadata["size_xyz"])
        size_x, size_y, size_z = metadata["size_xyz"]
        nodule_rows.append(
            {
                "seriesuid": seriesuid,
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "rel_z": rel_z,
                "rel_y": rel_y,
                "rel_x": rel_x,
                "w": float(row["w"]),
                "h": float(row["h"]),
                "d": float(row["d"]),
                "size_x": int(size_x),
                "size_y": int(size_y),
                "size_z": int(size_z),
                "log_connected_components": int(positive_scans.get(seriesuid, 0)),
            }
        )

    nodules_df = pd.DataFrame(nodule_rows)
    if nodules_df.empty:
        raise RuntimeError("No nodules found after joining log positives, labels CSV, and metadata.")

    count_df = (
        nodules_df.groupby("seriesuid")
        .size()
        .reset_index(name="nodule_count")
        .sort_values(["nodule_count", "seriesuid"], ascending=[False, True])
    )
    count_distribution = Counter(count_df["nodule_count"].astype(int).tolist())
    count_distribution_df = pd.DataFrame(
        [{"nodule_count": k, "scan_count": v} for k, v in sorted(count_distribution.items())]
    )
    count_distribution_df["probability"] = (
        count_distribution_df["scan_count"] / count_distribution_df["scan_count"].sum()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    nodules_csv = output_dir / "empirical_nodule_positions.csv"
    counts_csv = output_dir / "empirical_nodule_count_distribution.csv"
    npz_path = output_dir / "empirical_nodule_distribution.npz"
    json_path = output_dir / "empirical_nodule_distribution_summary.json"
    md_path = output_dir / "empirical_nodule_distribution_report.md"

    nodules_df.to_csv(nodules_csv, index=False)
    count_distribution_df.to_csv(counts_csv, index=False)
    np.savez_compressed(
        npz_path,
        relative_zyx=nodules_df[["rel_z", "rel_y", "rel_x"]].to_numpy(dtype=np.float32),
        voxel_zyx=nodules_df[["z", "y", "x"]].to_numpy(dtype=np.float32),
        nodule_counts=count_df["nodule_count"].to_numpy(dtype=np.int16),
        seriesuids=count_df["seriesuid"].to_numpy(dtype=str),
        count_values=count_distribution_df["nodule_count"].to_numpy(dtype=np.int16),
        count_probabilities=count_distribution_df["probability"].to_numpy(dtype=np.float32),
    )

    summary = {
        "source": source,
        "source_path": source_description,
        "log_path": str(log_path),
        "output_root": str(output_root),
        "labels_csv": str(labels_csv),
        "preprocessed_root": str(preprocessed_root),
        "source_scans": len(positive_scans),
        "positive_scans_with_position_rows": int(count_df.shape[0]),
        "nodule_positions": int(nodules_df.shape[0]),
        "missing_metadata_scans": sorted(set(missing_metadata)),
        "relative_position_mean_zyx": nodules_df[["rel_z", "rel_y", "rel_x"]].mean().round(6).to_dict(),
        "relative_position_std_zyx": nodules_df[["rel_z", "rel_y", "rel_x"]].std().round(6).to_dict(),
        "relative_position_quantiles_zyx": {
            q: nodules_df[["rel_z", "rel_y", "rel_x"]].quantile(float(q)).round(6).to_dict()
            for q in ["0.05", "0.25", "0.50", "0.75", "0.95"]
        },
        "nodule_count_distribution": {
            str(int(row["nodule_count"])): {
                "scan_count": int(row["scan_count"]),
                "probability": float(row["probability"]),
            }
            for _, row in count_distribution_df.iterrows()
        },
    }
    json_path.write_text(json.dumps(summary, indent=2))

    md = [
        "# Empirical LUNA16 Nodule Position Distribution",
        "",
        f"Source mode: `{source}`",
        f"Source path: `{source_description}`",
        f"Processed labels: `{labels_csv}`",
        f"Preprocessed root: `{preprocessed_root}`",
        "",
        "## Summary",
        "",
        f"- Scans found in source: **{summary['source_scans']}**",
        f"- Positive scans with coordinate rows: **{summary['positive_scans_with_position_rows']}**",
        f"- Nodule positions: **{summary['nodule_positions']}**",
        f"- Missing metadata scans: **{len(summary['missing_metadata_scans'])}**",
        "",
        "## Relative Position Distribution",
        "",
        "Coordinates are normalized to each preprocessed volume size and saved in `[rel_z, rel_y, rel_x]`.",
        "",
        "| Axis | Mean | Std | Q05 | Q25 | Q50 | Q75 | Q95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for axis in ["rel_z", "rel_y", "rel_x"]:
        md.append(
            "| {axis} | {mean:.3f} | {std:.3f} | {q05:.3f} | {q25:.3f} | {q50:.3f} | {q75:.3f} | {q95:.3f} |".format(
                axis=axis,
                mean=summary["relative_position_mean_zyx"][axis],
                std=summary["relative_position_std_zyx"][axis],
                q05=summary["relative_position_quantiles_zyx"]["0.05"][axis],
                q25=summary["relative_position_quantiles_zyx"]["0.25"][axis],
                q50=summary["relative_position_quantiles_zyx"]["0.50"][axis],
                q75=summary["relative_position_quantiles_zyx"]["0.75"][axis],
                q95=summary["relative_position_quantiles_zyx"]["0.95"][axis],
            )
        )
    md.extend(
        [
            "",
            "## Nodule Count Distribution Per Positive Scan",
            "",
            "| Nodules per scan | Scan count | Probability |",
            "|---:|---:|---:|",
        ]
    )
    for _, row in count_distribution_df.iterrows():
        md.append(f"| {int(row['nodule_count'])} | {int(row['scan_count'])} | {float(row['probability']):.4f} |")
    md.extend(
        [
            "",
            "## Files",
            "",
            f"- Nodule positions CSV: `{nodules_csv}`",
            f"- Count distribution CSV: `{counts_csv}`",
            f"- Sampling arrays NPZ: `{npz_path}`",
            f"- JSON summary: `{json_path}`",
            "",
            "Use `relative_zyx` as the empirical position pool and `count_values/count_probabilities` to sample the number of pseudo-nodules for scans without nodules.",
        ]
    )
    md_path.write_text("\n".join(md))
    return nodules_csv, counts_csv, md_path


def main():
    parser = argparse.ArgumentParser(
        description="Build empirical nodule position and count distributions from LUNA16 saliency logs."
    )
    parser.add_argument("--log-path", type=Path, default=Path("luna16_synthetic_2d.txt"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/luna16_synthetic_gt"),
        help="Root containing generated synthetic samples. Series UIDs are extracted from paths.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "output", "log", "control-points"),
        default="auto",
        help="Use generated output paths, saved control-point npy files, the saliency log, or output if present otherwise log.",
    )
    parser.add_argument("--labels-csv", type=Path, default=Path("data/LUNA16_preprocessed/luna16_labels.csv"))
    parser.add_argument("--preprocessed-root", type=Path, default=Path("data/LUNA16_preprocessed"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/luna16_nodule_position_distribution"),
    )
    args = parser.parse_args()
    if args.source == "control-points":
        positive_scans = parse_positive_scans_from_log(args.log_path)
        outputs = build_distribution_from_output_control_points(
            output_root=args.output_root,
            output_dir=args.output_dir,
            positive_scans=positive_scans,
        )
    else:
        outputs = build_distribution(
            log_path=args.log_path,
            output_root=args.output_root,
            source=args.source,
            labels_csv=args.labels_csv,
            preprocessed_root=args.preprocessed_root,
            output_dir=args.output_dir,
        )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
