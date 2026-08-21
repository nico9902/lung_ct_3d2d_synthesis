from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score


def metric_row(predictions: pd.DataFrame, pooling: str) -> dict[str, object]:
    y = predictions["label"].astype(int).to_numpy()
    pred = predictions["prediction"].astype(int).to_numpy()
    score = predictions["score"].astype(float).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "pooling": pooling,
        "samples": int(len(predictions)),
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
        "auc": float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else 0.0,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LUNA16 detection-MIL test predictions.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--poolings", nargs="+", default=["mean", "max", "attention"])
    parser.add_argument("--report-name", default="detection_mil_pooled_results.md")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = []
    for pooling in args.poolings:
        files = sorted(output_dir.glob(f"fold_*/{pooling}/test_predictions.csv"))
        if not files:
            continue
        predictions = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        predictions.to_csv(output_dir / f"all_test_predictions_{pooling}.csv", index=False)
        row = metric_row(predictions, pooling)
        row["folds"] = len(files)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No test_predictions.csv files found under {output_dir}")

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "pooled_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    with (output_dir / "pooled_metrics.json").open("w") as handle:
        json.dump(rows, handle, indent=2)

    lines = [
        "# LUNA16 Detection-Crop MIL Baseline",
        "",
        "Patient-level baseline that classifies bags of detector-centered crops without constructing an adaptive RBF/TPS surface.",
        "",
        "| Pooling | Folds | Samples | AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pooling']} | {row['folds']} | {row['samples']} | "
            f"{row['auc']:.4f} | {row['mcc']:.4f} | {row['accuracy']:.4f} | {row['f1']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this baseline answers whether the detector output alone, represented as independent candidate crops and pooled at patient level, is sufficient. A gap versus adaptive RBF supports the value of building a patient-level 2D surface that preserves spatial context and candidate interactions.",
            "",
        ]
    )
    (output_dir / args.report_name).write_text("\n".join(lines))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
