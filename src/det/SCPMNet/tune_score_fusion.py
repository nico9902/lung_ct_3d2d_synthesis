from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.det.SCPMNet.fp_reduction import evaluate_froc, ground_truth_by_series


def parse_grid(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune SCPMNet/classifier score fusion exponents for FROC.")
    parser.add_argument("--rescored-candidates", required=True, help="CSV from rescore_candidates.py containing scpm/classifier probabilities.")
    parser.add_argument("--csv-path", default="data/lidc_process/lidc_labels.csv")
    parser.add_argument("--data-root", default="data/lidc_process")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", default="outputs/scpmnet/fp_reduction/fusion_tuning")
    parser.add_argument("--scpm-exponents", default="0,0.25,0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--classifier-exponents", default="0,0.25,0.5,0.75,1,1.25,1.5,2,3")
    parser.add_argument(
        "--input-score-mode",
        choices=("auto", "raw", "multiply"),
        default="auto",
        help=(
            "How to interpret probability when scpm_probability is missing. "
            "'multiply' recovers SCPM as probability/classifier_probability; "
            "'raw' treats probability as SCPM."
        ),
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.rescored_candidates)
    if "scpm_probability" not in df.columns:
        if "classifier_probability" not in df.columns:
            raise ValueError("Input must contain classifier_probability, and ideally scpm_probability.")
        mode = args.input_score_mode
        if mode == "auto":
            mode = "multiply"
        if mode == "multiply":
            print("scpm_probability missing; recovering SCPM probability as probability/classifier_probability.")
            cls = np.clip(df["classifier_probability"].astype(float).to_numpy(), args.eps, None)
            df["scpm_probability"] = np.clip(df["probability"].astype(float).to_numpy() / cls, args.eps, 1.0)
        else:
            print("scpm_probability missing; using current probability as SCPM probability.")
            df["scpm_probability"] = df["probability"].astype(float)
    if "classifier_probability" not in df.columns:
        raise ValueError("Input must contain classifier_probability.")

    gt = ground_truth_by_series(args.csv_path, args.split, args.data_root, skip_missing_images=False)
    scpm_exponents = parse_grid(args.scpm_exponents)
    classifier_exponents = parse_grid(args.classifier_exponents)
    base_scpm = np.clip(df["scpm_probability"].to_numpy(dtype=float), args.eps, 1.0)
    base_cls = np.clip(df["classifier_probability"].to_numpy(dtype=float), args.eps, 1.0)

    rows = []
    best = None
    best_df = None
    best_froc = None
    best_curve = None
    for a in scpm_exponents:
        for b in classifier_exponents:
            print(f"Evaluating fusion with SCPM exponent {a} and classifier exponent {b}...")
            if a == 0.0 and b == 0.0:
                continue
            tuned = df.copy()
            tuned["probability"] = np.power(base_scpm, a) * np.power(base_cls, b)
            froc, curve, mean_froc = evaluate_froc(tuned, gt, score_col="probability")
            one_fp = float(froc.loc[np.isclose(froc["fp_per_scan"], 1.0), "sensitivity"].iloc[0])
            rows.append(
                {
                    "scpm_exponent": a,
                    "classifier_exponent": b,
                    "mean_froc": mean_froc,
                    "sensitivity_1fp": one_fp,
                    **{f"sens_{rate:g}fp": value for rate, value in zip(froc["fp_per_scan"], froc["sensitivity"])},
                }
            )
            if best is None or one_fp > best["sensitivity_1fp"] or (one_fp == best["sensitivity_1fp"] and mean_froc > best["mean_froc"]):
                best = rows[-1]
                best_df = tuned
                best_froc = froc
                best_curve = curve

    results = pd.DataFrame(rows).sort_values(["sensitivity_1fp", "mean_froc"], ascending=False)
    results_path = output_dir / f"{args.split}_fusion_grid.csv"
    results.to_csv(results_path, index=False)

    assert best is not None and best_df is not None and best_froc is not None and best_curve is not None
    best_predictions_path = output_dir / f"{args.split}_predictions_fusion_best.csv"
    best_froc_path = output_dir / f"{args.split}_froc_fusion_best.csv"
    best_curve_path = output_dir / f"{args.split}_froc_curve_fusion_best.csv"
    best_df.to_csv(best_predictions_path, index=False)
    best_froc.to_csv(best_froc_path, index=False)
    best_curve.to_csv(best_curve_path, index=False)

    print(f"Wrote grid to {results_path}")
    print("Best:")
    print(pd.DataFrame([best]).to_string(index=False))
    print(f"Wrote best predictions to {best_predictions_path}")
    print(f"Wrote best FROC to {best_froc_path}")


if __name__ == "__main__":
    main()
