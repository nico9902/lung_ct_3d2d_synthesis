"""Export CPMNetv2 test boxes for adaptive synthesis and frozen 3DINO probes.

Coordinates and sizes remain in voxels of the processed CT grid. No score
filtering is applied here; downstream experiments select their own candidates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_predictions(rows: pd.DataFrame) -> pd.DataFrame:
    required = ["seriesuid", "coordX", "coordY", "coordZ", "probability", "w", "h", "d"]
    missing = set(required) - set(rows.columns)
    if missing:
        raise ValueError(f"Missing detector columns: {sorted(missing)}")
    if rows.seriesuid.isna().any():
        raise ValueError("Missing seriesuid in detector predictions")
    values = rows[required[1:]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Nonfinite detector coordinates, scores or sizes")
    if (values[["w", "h", "d"]] <= 0).any().any() or not values.probability.between(0, 1).all():
        raise ValueError("Detector sizes must be positive and scores in [0, 1]")
    result = values[["coordZ", "coordY", "coordX"]].copy()
    result.insert(0, "seriesuid", rows.seriesuid.astype(str))
    result["radius"] = values[["w", "h", "d"]].max(axis=1) / 2
    result["probability"] = values.probability
    return result


def export_predictions(input_root: Path, output_root: Path, splits_dir: Path,
                       folds: list[int], pooled_csv: Path) -> None:
    if not folds or len(set(folds)) != len(folds) or any(fold not in range(10) for fold in folds):
        raise ValueError("Provide distinct fold indices in 0..9")
    exports, pooled, seen = [], [], set()
    for fold in folds:
        candidates = sorted(input_root.glob(f"*fold{fold}/predictions/test_predictions_test.csv"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one test prediction CSV for fold {fold}; found {candidates}")
        source = candidates[0]
        rows = normalize_predictions(pd.read_csv(source, dtype={"seriesuid": str}))
        split = pd.read_csv(splits_dir / f"luna16_fold{fold}.csv", dtype={"seriesuid": str})
        test_ids = set(split.loc[split.split == "test", "seriesuid"])
        if not test_ids:
            raise ValueError(f"No test scans in fold {fold}")
        if seen & test_ids:
            raise ValueError(f"Test scans occur in multiple folds: {sorted(seen & test_ids)}")
        seen.update(test_ids)
        unexpected = set(rows.seriesuid) - test_ids
        if unexpected:
            raise ValueError(f"Non-test detector predictions in fold {fold}: {sorted(unexpected)}")
        exports.append((output_root / source.parent.parent.name / "predictions/test_predictions.csv", rows))
        probe = rows.rename(columns={"coordX": "x", "coordY": "y", "coordZ": "z", "probability": "prob_nodule"})
        probe.insert(1, "fold", fold)
        pooled.append(probe)
        print(f"fold {fold}: {len(rows)} detections; {len(test_ids - set(rows.seriesuid))} scans without candidates")
    # Validate all folds before writing any output.
    for destination, rows in exports:
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(destination, index=False)
    pooled_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(pooled, ignore_index=True).to_csv(pooled_csv, index=False)
    print(f"Wrote synthesis predictions to {output_root} and pooled candidates to {pooled_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/cpmnetv2_luna16_10fold_bf16_guarded"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cpmnetv2_luna16_10fold_bf16_guarded_results/normalized_predictions"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/processed/cv_splits"))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--pooled-csv", type=Path, default=Path("outputs/cpmnetv2_paper_detector_all_oof_predictions.csv"))
    args = parser.parse_args()
    export_predictions(args.input_root, args.output_root, args.splits_dir, args.folds, args.pooled_csv)


if __name__ == "__main__":
    main()
