from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_FP_RATES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def fold_index(path: Path) -> int:
    match = re.search(r"fold(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot infer fold index from {path}")
    return int(match.group(1))


def gt_from_rows(rows: pd.DataFrame) -> np.ndarray:
    valid = rows.copy()
    if "label" in valid.columns:
        valid = valid[valid["label"].astype(str).str.lower().isin(("nodule", "1", "true", "positive"))]
    if valid.empty:
        return np.zeros((0, 4), dtype=np.float32)

    if all(col in valid.columns for col in ("coordZ", "coordY", "coordX")):
        coord_cols = ("coordZ", "coordY", "coordX")
    elif all(col in valid.columns for col in ("z", "y", "x")):
        coord_cols = ("z", "y", "x")
    else:
        return np.zeros((0, 4), dtype=np.float32)

    valid = valid.dropna(subset=list(coord_cols))
    spheres: list[list[float]] = []
    for _, row in valid.iterrows():
        if "radius" in row and pd.notna(row["radius"]):
            radius = float(row["radius"])
        elif "diameter_mm" in row and pd.notna(row["diameter_mm"]):
            radius = float(row["diameter_mm"]) / 2.0
        else:
            dims = [float(row[col]) for col in ("w", "h", "d") if col in row and pd.notna(row[col])]
            radius = max(dims) / 2.0 if len(dims) == 3 else 0.0
        if radius > 0:
            spheres.append([float(row[coord_cols[0]]), float(row[coord_cols[1]]), float(row[coord_cols[2]]), radius])
    return np.asarray(spheres, dtype=np.float32) if spheres else np.zeros((0, 4), dtype=np.float32)


def evaluate_froc(
    pred_df: pd.DataFrame,
    gt_by_series: dict[str, np.ndarray],
    fp_rates: Sequence[float] = DEFAULT_FP_RATES,
    score_col: str = "probability",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    total_gt = int(sum(len(v) for v in gt_by_series.values()))
    num_scans = int(len(gt_by_series))
    if total_gt == 0 or num_scans == 0:
        raise ValueError("Cannot evaluate FROC without ground-truth scans and nodules.")

    if pred_df.empty:
        fp_per_scan = np.asarray([0.0], dtype=np.float32)
        sensitivity = np.asarray([0.0], dtype=np.float32)
    else:
        pred_df = pred_df.sort_values(score_col, ascending=False).reset_index(drop=True)
        matched = {seriesuid: np.zeros(len(gt), dtype=bool) for seriesuid, gt in gt_by_series.items()}
        tp = 0
        fp = 0
        sensitivities = [0.0]
        fps = [0.0]
        for _, row in pred_df.iterrows():
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
                fp += 1
            else:
                matched[seriesuid][match_idx] = True
                tp += 1
            sensitivities.append(tp / total_gt)
            fps.append(fp / num_scans)
        fp_per_scan = np.asarray(fps, dtype=np.float32)
        sensitivity = np.maximum.accumulate(np.asarray(sensitivities, dtype=np.float32))

    rows = []
    values = []
    for rate in fp_rates:
        eligible = sensitivity[fp_per_scan <= float(rate)]
        value = float(eligible.max()) if eligible.size else 0.0
        values.append(value)
        rows.append({"fp_per_scan": float(rate), "sensitivity": value})
    return pd.DataFrame(rows), pd.DataFrame({"fp_per_scan": fp_per_scan, "sensitivity": sensitivity}), float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SCPMNet LIDC 10-fold prediction/FROC outputs.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/scpmnet_lidc_10fold"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/lidc_process/cv_splits"))
    parser.add_argument("--fold-glob", default="scpmnet_paper_lidc_fold*")
    parser.add_argument("--score-col", default="probability")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    fold_dirs = sorted(args.output_root.glob(args.fold_glob), key=fold_index)
    if len(fold_dirs) != 10:
        raise ValueError(f"Expected 10 fold directories, found {len(fold_dirs)} in {args.output_root}")

    out_dir = args.out_dir or args.output_root / "cv_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    all_fold_frocs = []
    all_predictions = []
    gt_by_series: dict[str, np.ndarray] = {}

    for fold_dir in fold_dirs:
        fold = fold_index(fold_dir)
        froc_path = fold_dir / "predictions" / "test_froc.csv"
        pred_path = fold_dir / "predictions" / "test_predictions.csv"
        split_path = args.split_dir / f"lidc_fold{fold}.csv"

        froc = pd.read_csv(froc_path)
        fold_row = {"fold": fold, "mean_froc": float(froc["sensitivity"].mean())}
        for _, row in froc.iterrows():
            fold_row[f"froc_{row.fp_per_scan:g}fp"] = float(row.sensitivity)
        fold_rows.append(fold_row)
        fold_froc = froc.copy()
        fold_froc["fold"] = fold
        all_fold_frocs.append(fold_froc)

        pred = pd.read_csv(pred_path)
        pred["fold"] = fold
        all_predictions.append(pred)

        split = pd.read_csv(split_path)
        test_rows = split[split["split"].astype(str) == "test"]
        for seriesuid, rows in test_rows.groupby("seriesuid", sort=False):
            gt_by_series[str(seriesuid)] = gt_from_rows(rows)

    fold_summary = pd.DataFrame(fold_rows).sort_values("fold")
    fold_summary.to_csv(out_dir / "fold_summary.csv", index=False)

    per_rate = (
        pd.concat(all_fold_frocs, ignore_index=True)
        .groupby("fp_per_scan")["sensitivity"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    per_rate.to_csv(out_dir / "fold_mean_froc_by_fp_rate.csv", index=False)

    pooled_predictions = pd.concat(all_predictions, ignore_index=True)
    pooled_predictions.to_csv(out_dir / "pooled_test_predictions.csv", index=False)
    pooled_eval = pooled_predictions[["seriesuid", "coordZ", "coordY", "coordX", "radius", args.score_col]]
    if args.score_col != "probability":
        pooled_eval = pooled_eval.rename(columns={args.score_col: "probability"})
    pooled_froc, pooled_curve, pooled_mean = evaluate_froc(pooled_eval, gt_by_series)
    pooled_froc.to_csv(out_dir / "pooled_test_froc.csv", index=False)
    pooled_curve.to_csv(out_dir / "pooled_test_froc_curve.csv", index=False)

    print(f"Wrote aggregate files to {out_dir}")
    print(f"Mean over fold mean FROC: {fold_summary['mean_froc'].mean():.6f} +/- {fold_summary['mean_froc'].std(ddof=1):.6f}")
    print(f"Pooled mean FROC: {pooled_mean:.6f}")
    print(f"Pooled scans: {len(gt_by_series)} | pooled GT nodules: {sum(len(v) for v in gt_by_series.values())}")


if __name__ == "__main__":
    main()
