"""Aggregate partial-fine-tuning 3DINO-ViT results across all 10 LUNA16 folds.

Uses the shared representation-agnostic pooled/per-fold metric utilities.
There is a single representation here (mean+max window pooling).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.luna16_foundation_common import (  # noqa: E402
    PER_FOLD_METRIC_KEYS,
    per_fold_summary,
    pooled_metric_row,
)

REPRESENTATION = "3dino_partial_ft"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LUNA16 3DINO partial-fine-tuning results across folds.")
    parser.add_argument("--output-dir", required=True, help="Directory containing fold_*/3dino_partial_ft/ subdirs.")
    parser.add_argument("--report-name", default="dino3d_pooled_results.md")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    files = sorted(output_dir.glob("fold_*/3dino_partial_ft/test_predictions.csv"))
    if not files:
        raise RuntimeError(f"No test_predictions.csv files found under {output_dir}")

    predictions = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    predictions.to_csv(output_dir / f"all_test_predictions_{REPRESENTATION}.csv", index=False)

    pooled_row = pooled_metric_row(predictions, REPRESENTATION)
    pooled_row["folds"] = len(files)
    pd.DataFrame([pooled_row]).to_csv(output_dir / "pooled_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    with (output_dir / "pooled_metrics.json").open("w") as handle:
        json.dump([pooled_row], handle, indent=2)

    fold_df = per_fold_summary(output_dir, REPRESENTATION)
    fold_df.to_csv(output_dir / f"per_fold_metrics_{REPRESENTATION}.csv", index=False)

    mean_std_row = {"representation": REPRESENTATION, "folds": len(fold_df)}
    for key in PER_FOLD_METRIC_KEYS:
        mean_std_row[f"{key}_mean"] = float(fold_df[key].mean())
        mean_std_row[f"{key}_std"] = float(fold_df[key].std(ddof=1)) if len(fold_df) > 1 else 0.0
    mean_std_df = pd.DataFrame([mean_std_row])
    mean_std_df.to_csv(output_dir / "per_fold_mean_std_metrics.csv", index=False)

    lines = [
        "# LUNA16 3DINO-ViT Partial-Fine-Tuning Baseline",
        "",
        "3DINO-ViT (ViT-Large-3D, official AICONSlab checkpoint) with the first 12 of 24 transformer blocks "
        "frozen and the second 12 blocks + final norm fine-tuned, evaluated on the exact same 10 LUNA16 "
        "patient-level malignancy folds used by the Adaptive RBF and Rad-JEPA-3D experiments. "
        "No lesion-guided information (detections, nodule coordinates, masks) is used -- only overlapping "
        "112^3 sliding windows over the full CT volume, aggregated via mean+max pooling.",
        "",
        "## Pooled out-of-fold test metrics (all 10 folds concatenated)",
        "",
        "| Representation | Folds | Samples | AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {pooled_row['representation']} | {pooled_row['folds']} | {pooled_row['samples']} | "
        f"{pooled_row['auc']:.4f} | {pooled_row['mcc']:.4f} | {pooled_row['accuracy']:.4f} | {pooled_row['f1']:.4f} | "
        f"{pooled_row['precision']:.4f} | {pooled_row['recall']:.4f} | {pooled_row['tn']} | {pooled_row['fp']} | "
        f"{pooled_row['fn']} | {pooled_row['tp']} |",
        "",
        "## Per-fold mean +/- std (ddof=1) test metrics",
        "",
        "| Representation | Folds | AUC | MCC | Accuracy | F1 | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {mean_std_row['representation']} | {mean_std_row['folds']} | "
        f"{mean_std_row['auc_mean']:.4f} +/- {mean_std_row['auc_std']:.4f} | "
        f"{mean_std_row['mcc_mean']:.4f} +/- {mean_std_row['mcc_std']:.4f} | "
        f"{mean_std_row['acc_mean']:.4f} +/- {mean_std_row['acc_std']:.4f} | "
        f"{mean_std_row['f1_mean']:.4f} +/- {mean_std_row['f1_std']:.4f} | "
        f"{mean_std_row['precision_mean']:.4f} +/- {mean_std_row['precision_std']:.4f} | "
        f"{mean_std_row['recall_mean']:.4f} +/- {mean_std_row['recall_std']:.4f} |",
        "",
    ]
    (output_dir / args.report_name).write_text("\n".join(lines))

    print(pd.DataFrame([pooled_row]).to_string(index=False))
    print()
    print(mean_std_df.to_string(index=False))


if __name__ == "__main__":
    main()
