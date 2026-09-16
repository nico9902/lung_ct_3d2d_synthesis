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


def metric_row(predictions: pd.DataFrame, backbone: str) -> dict[str, object]:
    y = predictions["label"].astype(int).to_numpy()
    pred = predictions["prediction"].astype(int).to_numpy()
    score = predictions["score"].astype(float).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "backbone": backbone,
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
    parser = argparse.ArgumentParser(description="Aggregate 2.5D slice-attention test predictions across LUNA16 folds.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--backbones", nargs="+", default=["efficientnet_v2_s"])
    parser.add_argument("--name", default="luna16_slice_attention_2p5d_pooled_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for backbone in args.backbones:
        files = sorted(output_dir.glob(f"fold_*/{backbone}/test_predictions.csv"))
        if not files:
            continue
        predictions = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        predictions.to_csv(output_dir / f"all_test_predictions_{backbone}.csv", index=False)
        row = metric_row(predictions, backbone)
        row["folds"] = len(files)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No test_predictions.csv files found under {output_dir}")

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "pooled_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    with (output_dir / "pooled_metrics.json").open("w") as handle:
        json.dump(rows, handle, indent=2)

    lines = [
        "# LUNA16 2.5D Slice-Attention Baseline",
        "",
        "Non-adaptive patient-level classifier over all axial slices from the preprocessed LUNA16 volume.",
        "",
        "| Backbone | Folds | Samples | AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['backbone']} | {row['folds']} | {row['samples']} | "
            f"{row['auc']:.4f} | {row['mcc']:.4f} | {row['accuracy']:.4f} | {row['f1']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this baseline tests whether a pretrained 2D encoder plus learned soft attention over native axial context can match the adaptive detector-guided 2D surface.",
            "",
        ]
    )
    report_path = results_dir / f"{args.name}.md"
    report_path.write_text("\n".join(lines))
    print(summary.to_string(index=False))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
