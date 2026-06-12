from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


POSITIVE_LABELS = {"nodule", "1", "true", "positive"}


def row_radius(row: pd.Series) -> float:
    if "radius" in row and pd.notna(row["radius"]):
        return float(row["radius"])
    for diameter_col in ("diameter", "diameter_mm"):
        if diameter_col in row and pd.notna(row[diameter_col]):
            return float(row[diameter_col]) / 2.0
    depth_col = "depth" if "depth" in row and pd.notna(row.get("depth")) else "d"
    dims = []
    for col in ("w", "h", depth_col):
        if col in row and pd.notna(row[col]):
            dims.append(float(row[col]))
    return max(dims) / 2.0 if len(dims) == 3 else 0.0


def load_spheres(csv_path: Path, split: str | None, min_radius: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if split:
        if "split" not in df.columns:
            raise ValueError(f"Requested split={split!r}, but {csv_path} has no split column.")
        df = df[df["split"].astype(str) == split].copy()

    if "seriesuid" not in df.columns:
        raise ValueError(f"{csv_path} must contain a seriesuid column.")

    if all(col in df.columns for col in ("coordZ", "coordY", "coordX")):
        coord_cols = ("coordZ", "coordY", "coordX")
    elif all(col in df.columns for col in ("z", "y", "x")):
        coord_cols = ("z", "y", "x")
    else:
        raise ValueError(f"{csv_path} must contain either coordZ/coordY/coordX or z/y/x columns.")

    valid = df.dropna(subset=list(coord_cols)).copy()
    if "label" in valid.columns:
        valid = valid[valid["label"].astype(str).str.lower().isin(POSITIVE_LABELS)].copy()

    valid["radius"] = valid.apply(row_radius, axis=1)
    valid = valid[valid["radius"] >= min_radius].copy()
    valid = valid.reset_index(drop=False).rename(columns={"index": "source_row"})
    valid["gt_index"] = np.arange(len(valid), dtype=np.int64)
    valid["coordZ"] = valid[coord_cols[0]].astype(float)
    valid["coordY"] = valid[coord_cols[1]].astype(float)
    valid["coordX"] = valid[coord_cols[2]].astype(float)
    return valid


def sphere_iou(distance: float, r1: float, r2: float) -> float:
    r1 = max(float(r1), 1e-9)
    r2 = max(float(r2), 1e-9)
    d = max(float(distance), 0.0)
    vol1 = 4.0 / 3.0 * np.pi * r1**3
    vol2 = 4.0 / 3.0 * np.pi * r2**3

    if d >= r1 + r2:
        inter = 0.0
    elif d + min(r1, r2) <= max(r1, r2):
        inter = 4.0 / 3.0 * np.pi * min(r1, r2) ** 3
    else:
        cos1 = np.clip((r1**2 + d**2 - r2**2) / (2 * r1 * d), -0.999999, 0.999999)
        cos2 = np.clip((r2**2 + d**2 - r1**2) / (2 * r2 * d), -0.999999, 0.999999)
        h1 = r1 * (1.0 - cos1)
        h2 = r2 * (1.0 - cos2)
        inter = np.pi * h1**2 * (r1 - h1 / 3.0) + np.pi * h2**2 * (r2 - h2 / 3.0)
    return float(inter / max(vol1 + vol2 - inter, 1e-9))


def diagnose_pairs(spheres: pd.DataFrame, near_margin: float) -> pd.DataFrame:
    rows = []
    for seriesuid, group in spheres.groupby("seriesuid", sort=False):
        group = group.reset_index(drop=True)
        centers = group[["coordZ", "coordY", "coordX"]].to_numpy(dtype=float)
        radii = group["radius"].to_numpy(dtype=float)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                distance = float(np.linalg.norm(centers[i] - centers[j]))
                r1 = float(radii[i])
                r2 = float(radii[j])
                overlap_margin = r1 + r2 - distance
                overlap = overlap_margin > 0.0
                containment_margin = max(r1, r2) - (distance + min(r1, r2))
                contained = containment_margin >= 0.0
                near_or_touching = distance <= (r1 + r2 + near_margin)
                if not (overlap or near_or_touching):
                    continue

                larger_idx = i if r1 >= r2 else j
                smaller_idx = j if larger_idx == i else i
                rows.append(
                    {
                        "seriesuid": seriesuid,
                        "gt_index_a": int(group.loc[i, "gt_index"]),
                        "gt_index_b": int(group.loc[j, "gt_index"]),
                        "source_row_a": int(group.loc[i, "source_row"]),
                        "source_row_b": int(group.loc[j, "source_row"]),
                        "za": centers[i, 0],
                        "ya": centers[i, 1],
                        "xa": centers[i, 2],
                        "ra": r1,
                        "zb": centers[j, 0],
                        "yb": centers[j, 1],
                        "xb": centers[j, 2],
                        "rb": r2,
                        "distance": distance,
                        "overlap": bool(overlap),
                        "overlap_margin": overlap_margin,
                        "contained": bool(contained),
                        "containment_margin": containment_margin,
                        "near_margin": near_margin,
                        "sphere_iou": sphere_iou(distance, r1, r2),
                        "larger_gt_index": int(group.loc[larger_idx, "gt_index"]),
                        "smaller_gt_index": int(group.loc[smaller_idx, "gt_index"]),
                        "larger_radius": float(radii[larger_idx]),
                        "smaller_radius": float(radii[smaller_idx]),
                    }
                )
    return pd.DataFrame(rows)


def summarize(spheres: pd.DataFrame, pairs: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if pairs.empty:
        flagged_gt = set()
        contained_gt = set()
        overlapping_gt = set()
    else:
        flagged_gt = set(pairs["gt_index_a"]).union(set(pairs["gt_index_b"]))
        contained_gt = set(pairs.loc[pairs["contained"], "smaller_gt_index"])
        overlapping_gt = set(pairs.loc[pairs["overlap"], "gt_index_a"]).union(set(pairs.loc[pairs["overlap"], "gt_index_b"]))

    per_series_rows = []
    for seriesuid, group in spheres.groupby("seriesuid", sort=False):
        gt_indices = set(group["gt_index"].astype(int).tolist())
        series_pairs = pairs[pairs["seriesuid"] == seriesuid] if not pairs.empty else pd.DataFrame()
        per_series_rows.append(
            {
                "seriesuid": seriesuid,
                "num_nodules": len(group),
                "num_flagged_pairs": len(series_pairs),
                "num_overlap_pairs": int(series_pairs["overlap"].sum()) if not series_pairs.empty else 0,
                "num_containment_pairs": int(series_pairs["contained"].sum()) if not series_pairs.empty else 0,
                "num_flagged_nodules": len(gt_indices & flagged_gt),
                "num_overlapping_nodules": len(gt_indices & overlapping_gt),
                "num_contained_nodules": len(gt_indices & contained_gt),
            }
        )
    per_series = pd.DataFrame(per_series_rows).sort_values(
        ["num_containment_pairs", "num_overlap_pairs", "num_nodules"],
        ascending=False,
    )

    summary = {
        "num_series": int(spheres["seriesuid"].nunique()),
        "num_nodules": int(len(spheres)),
        "num_series_with_multiple_nodules": int((spheres.groupby("seriesuid").size() > 1).sum()),
        "num_flagged_pairs": int(len(pairs)),
        "num_overlap_pairs": int(pairs["overlap"].sum()) if not pairs.empty else 0,
        "num_containment_pairs": int(pairs["contained"].sum()) if not pairs.empty else 0,
        "num_flagged_nodules": int(len(flagged_gt)),
        "num_overlapping_nodules": int(len(overlapping_gt)),
        "num_contained_nodules": int(len(contained_gt)),
        "max_nodules_per_series": int(spheres.groupby("seriesuid").size().max()) if len(spheres) else 0,
    }
    return summary, per_series


def source_rows_to_remove(pairs: pd.DataFrame, policy: str) -> set[int]:
    if pairs.empty:
        return set()

    if policy == "contained_smaller":
        selected = pairs[pairs["contained"]].copy()
    elif policy == "overlap_smaller":
        selected = pairs[pairs["overlap"]].copy()
    elif policy == "flagged_smaller":
        selected = pairs.copy()
    else:
        raise ValueError(f"Unsupported clean policy: {policy}")

    rows: set[int] = set()
    for _, row in selected.iterrows():
        if int(row["smaller_gt_index"]) == int(row["gt_index_a"]):
            rows.add(int(row["source_row_a"]))
        else:
            rows.add(int(row["source_row_b"]))
    return rows


def write_clean_csv(
    input_csv_path: Path,
    output_csv_path: Path,
    pairs: pd.DataFrame,
    policy: str,
) -> dict:
    original = pd.read_csv(input_csv_path)
    remove_rows = source_rows_to_remove(pairs, policy)
    keep_mask = ~original.index.isin(remove_rows)
    cleaned = original[keep_mask].copy()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_csv_path, index=False)
    return {
        "clean_policy": policy,
        "clean_csv_path": str(output_csv_path),
        "removed_rows": len(remove_rows),
        "original_rows": int(len(original)),
        "clean_rows": int(len(cleaned)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose overlapping or contained SCPMNet/LIDC sphere labels.")
    parser.add_argument("--csv-path", default="data/lidc_process/lidc_labels.csv", help="Input label CSV.")
    parser.add_argument("--split", default=None, help="Optional split filter, e.g. train, val, or test.")
    parser.add_argument("--output-dir", default="outputs/scpmnet/dataset_diagnostics", help="Directory for output CSV/JSON files.")
    parser.add_argument("--near-margin", type=float, default=2.0, help="Also report pairs within this many voxels of touching.")
    parser.add_argument("--min-radius", type=float, default=0.0, help="Ignore nodules with radius smaller than this value.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of highest-risk pairs to print.")
    parser.add_argument("--write-clean-csv", action="store_true", help="Write a copy of the input labels with selected smaller nodules removed.")
    parser.add_argument(
        "--clean-policy",
        choices=("contained_smaller", "overlap_smaller", "flagged_smaller"),
        default="contained_smaller",
        help=(
            "contained_smaller removes only smaller nodules fully inside larger spheres; "
            "overlap_smaller removes smaller nodules from all overlapping pairs; "
            "flagged_smaller removes smaller nodules from overlapping and near-touching pairs."
        ),
    )
    parser.add_argument("--clean-csv-path", default=None, help="Optional output path for the cleaned CSV.")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spheres = load_spheres(csv_path, split=args.split, min_radius=args.min_radius)
    pairs = diagnose_pairs(spheres, near_margin=args.near_margin)
    summary, per_series = summarize(spheres, pairs)
    summary.update(
        {
            "csv_path": str(csv_path),
            "split": args.split,
            "near_margin": args.near_margin,
            "min_radius": args.min_radius,
        }
    )

    suffix = f"_{args.split}" if args.split else "_all"
    spheres_path = output_dir / f"nodules{suffix}.csv"
    pairs_path = output_dir / f"overlap_pairs{suffix}.csv"
    per_series_path = output_dir / f"per_series_summary{suffix}.csv"
    summary_path = output_dir / f"summary{suffix}.json"

    spheres[
        ["gt_index", "source_row", "seriesuid", "split", "coordZ", "coordY", "coordX", "radius"]
        if "split" in spheres.columns
        else ["gt_index", "source_row", "seriesuid", "coordZ", "coordY", "coordX", "radius"]
    ].to_csv(spheres_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    per_series.to_csv(per_series_path, index=False)

    if args.write_clean_csv:
        clean_csv_path = (
            Path(args.clean_csv_path)
            if args.clean_csv_path
            else output_dir / f"{csv_path.stem}_clean_{args.clean_policy}{suffix}{csv_path.suffix}"
        )
        summary["clean_csv"] = write_clean_csv(
            input_csv_path=csv_path,
            output_csv_path=clean_csv_path,
            pairs=pairs,
            policy=args.clean_policy,
        )

    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nWrote:\n  {spheres_path}\n  {pairs_path}\n  {per_series_path}\n  {summary_path}")

    if pairs.empty:
        print("\nNo overlapping or near-touching pairs found.")
        return

    print(f"\nTop {min(args.top_k, len(pairs))} pairs by containment, overlap margin, and IoU:")
    top = pairs.sort_values(
        ["contained", "overlap_margin", "sphere_iou"],
        ascending=False,
    ).head(args.top_k)
    cols = [
        "seriesuid",
        "gt_index_a",
        "gt_index_b",
        "distance",
        "ra",
        "rb",
        "overlap",
        "overlap_margin",
        "contained",
        "containment_margin",
        "sphere_iou",
    ]
    print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
