from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LUNA16 2.5D slice-attention test predictions.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument("--report-name", default="slice_attention_2p5d_pooled_results.md")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    files = sorted(output_dir.glob(f"fold_*/{args.backbone}/test_predictions.csv"))
    if not files:
        raise RuntimeError(f"No test_predictions.csv files found under {output_dir}")

    predictions = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    predictions.to_csv(output_dir / "all_test_predictions.csv", index=False)
    y = predictions["label"].astype(int).to_numpy()
    pred = predictions["prediction"].astype(int).to_numpy()
    score = predictions["score"].astype(float).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    row = {
        "folds": len(files),
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
        "mean_slice_count": float(predictions["slice_count"].mean()) if "slice_count" in predictions else None,
    }
    pd.DataFrame([row]).to_csv(output_dir / "pooled_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    with (output_dir / "pooled_metrics.json").open("w") as handle:
        json.dump(row, handle, indent=2)

    lines = [
        "# LUNA16 2.5D Slice-Attention Baseline",
        "",
        "Patient-level baseline using all axial slices from each preprocessed volume.",
        "",
        "| Backbone | Folds | Samples | AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {args.backbone} | {row['folds']} | {row['samples']} | {row['auc']:.4f} | {row['mcc']:.4f} | "
            f"{row['accuracy']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} |"
        ),
        "",
        f"Mean slice count per scan: `{row['mean_slice_count']:.2f}`." if row["mean_slice_count"] is not None else "",
        "",
    ]
    (output_dir / args.report_name).write_text("\n".join(lines))
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
