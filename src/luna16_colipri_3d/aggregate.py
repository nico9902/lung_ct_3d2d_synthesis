"""Aggregate frozen-COLIPRI linear-probe results across all 10 LUNA16 folds.

For each representation ("pooled", "dense_maxpool") this:
  - concatenates the out-of-fold ``test_predictions.csv`` from every fold into
    one pooled table and computes pooled AUC/MCC/F1/accuracy/precision/recall
    /confusion-matrix (same sklearn definitions as
    ``src/luna16_volume_3d/train_resnet18.py`` and
    ``src/luna16_detection_mil/aggregate.py``);
  - reads each fold's ``test_metrics.json`` and reports the mean/std (ddof=1)
    of each metric across folds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PER_FOLD_METRIC_KEYS = ["auc", "mcc", "acc", "f1", "precision", "recall"]


def pooled_metric_row(predictions: pd.DataFrame, representation: str) -> dict[str, object]:
    y = predictions["label"].astype(int).to_numpy()
    pred = predictions["prediction"].astype(int).to_numpy()
    score = predictions["score"].astype(float).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "representation": representation,
        "samples": int(len(predictions)),
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
        "auc": float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def per_fold_summary(output_dir: Path, representation: str) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(output_dir.glob(f"fold_*/{representation}/test_metrics.json")):
        fold = int(metrics_path.parent.parent.name.split("_")[1])
        with metrics_path.open() as handle:
            metrics = json.load(handle)
        row = {"fold": fold}
        for key in PER_FOLD_METRIC_KEYS:
            row[key] = metrics[f"test_{key}"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LUNA16 COLIPRI linear-probe results across folds.")
    parser.add_argument("--output-dir", required=True, help="Directory containing fold_*/{representation}/ subdirs.")
    parser.add_argument("--representations", nargs="+", default=["pooled", "dense_maxpool"])
    parser.add_argument("--report-name", default="colipri_pooled_results.md")
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
        "# LUNA16 COLIPRI-CRM Frozen Linear-Probe Baseline",
        "",
        "Frozen COLIPRI-CRM 3D foundation-model encoder (official checkpoint, official 192x192x192 "
        "@ 2mm-isotropic preprocessing) with a trained `Linear(768, 1)` probe, evaluated on the exact "
        "same 10 LUNA16 patient-level malignancy folds used by the Adaptive RBF experiments.",
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
