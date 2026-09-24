"""Aggregate frozen-RadJEPA-3D linear-probe results across all 10 LUNA16 folds.

Uses the shared representation-agnostic pooled/per-fold metric utilities and
customizes the report text for the four RadJEPA representations.
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

REPRESENTATIONS = ["resize32", "sliding_mean", "sliding_max", "sliding_mean_max"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LUNA16 RadJEPA-3D linear-probe results across folds.")
    parser.add_argument("--output-dir", required=True, help="Directory containing fold_*/{representation}/ subdirs.")
    parser.add_argument("--representations", nargs="+", default=REPRESENTATIONS)
    parser.add_argument("--report-name", default="radjepa_pooled_results.md")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pooled_rows = []
    fold_summaries: dict[str, pd.DataFrame] = {}

    for representation in args.representations:
        files = sorted(output_dir.glob(f"fold_*/{representation}/test_predictions.csv"))
        if not files:
            print(f"Skipping '{representation}': no test_predictions.csv found under {output_dir}")
            continue
        predictions = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        predictions.to_csv(output_dir / f"all_test_predictions_{representation}.csv", index=False)

        row = pooled_metric_row(predictions, representation)
        row["folds"] = len(files)
        pooled_rows.append(row)

        fold_df = per_fold_summary(output_dir, representation)
        fold_df.to_csv(output_dir / f"per_fold_metrics_{representation}.csv", index=False)
        fold_summaries[representation] = fold_df

    if not pooled_rows:
        raise RuntimeError(f"No test_predictions.csv files found under {output_dir}")

    pooled_summary = pd.DataFrame(pooled_rows)
    pooled_summary.to_csv(output_dir / "pooled_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    with (output_dir / "pooled_metrics.json").open("w") as handle:
        json.dump(pooled_rows, handle, indent=2)

    mean_std_rows = []
    for representation, fold_df in fold_summaries.items():
        row = {"representation": representation, "folds": len(fold_df)}
        for key in PER_FOLD_METRIC_KEYS:
            row[f"{key}_mean"] = float(fold_df[key].mean())
            row[f"{key}_std"] = float(fold_df[key].std(ddof=1)) if len(fold_df) > 1 else 0.0
        mean_std_rows.append(row)
    mean_std_df = pd.DataFrame(mean_std_rows)
    mean_std_df.to_csv(output_dir / "per_fold_mean_std_metrics.csv", index=False)

    lines = [
        "# LUNA16 Rad-JEPA-3D Frozen Linear-Probe Baseline",
        "",
        "Frozen Rad-JEPA-3D (H-Mamba hybrid encoder) 3D foundation-model baseline with a trained "
        "`Linear(feature_dim, 1)` probe, evaluated on the exact same 10 LUNA16 patient-level malignancy "
        "folds used by the Adaptive RBF experiments. No lesion-guided information (detections, nodule "
        "coordinates, masks) is used -- Rad-JEPA sees the full CT volume only.",
        "",
        "## Pooled out-of-fold test metrics (all 10 folds concatenated)",
        "",
        "| Representation | Folds | Samples | AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled_rows:
        lines.append(
            f"| {row['representation']} | {row['folds']} | {row['samples']} | "
            f"{row['auc']:.4f} | {row['mcc']:.4f} | {row['accuracy']:.4f} | {row['f1']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} |"
        )

    lines.extend(
        [
            "",
            "## Per-fold mean +/- std (ddof=1) test metrics",
            "",
            "| Representation | Folds | AUC | MCC | Accuracy | F1 | Precision | Recall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mean_std_rows:
        lines.append(
            f"| {row['representation']} | {row['folds']} | "
            f"{row['auc_mean']:.4f} +/- {row['auc_std']:.4f} | "
            f"{row['mcc_mean']:.4f} +/- {row['mcc_std']:.4f} | "
            f"{row['acc_mean']:.4f} +/- {row['acc_std']:.4f} | "
            f"{row['f1_mean']:.4f} +/- {row['f1_std']:.4f} | "
            f"{row['precision_mean']:.4f} +/- {row['precision_std']:.4f} | "
            f"{row['recall_mean']:.4f} +/- {row['recall_std']:.4f} |"
        )
    lines.append("")
    (output_dir / args.report_name).write_text("\n".join(lines))

    print(pooled_summary.to_string(index=False))
    print()
    print(mean_std_df.to_string(index=False))


if __name__ == "__main__":
    main()
