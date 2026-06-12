import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PRED_PATH = "predictions_marco.csv"
GT_PATH = "data/processed/lidc_labels.csv"

FP_RATES = [0.125, 0.25, 0.5, 1, 2, 4, 8]

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_froc(pred_df, gt_df, fp_rates=FP_RATES):
    pred_df = pred_df.copy()
    gt_df = gt_df.copy()

    pred_df = pred_df.sort_values("score", ascending=False)

    scans = gt_df["seriesuid"].unique()
    num_scans = len(scans)
    num_gt = len(gt_df)

    results = []

    thresholds = sorted(pred_df["score"].unique(), reverse=True)

    for thr in thresholds:
        preds_thr = pred_df[pred_df["score"] >= thr]

        tp = 0
        fp = 0
        matched_gt = set()

        for _, pred in preds_thr.iterrows():
            pid = pred["patient_id"]
            pred_center = (pred["x_location"], pred["y_location"])

            gt_scan = gt_df[gt_df["seriesuid"] == pid]
            # select only GT nodules that are in the same slice (z) as the prediction
            gt_scan = gt_scan[gt_scan["z"] == pred["z_location"]]

            found_match = False
            best_gt_idx = None
            best_dist = np.inf

            for gt_idx, gt in gt_scan.iterrows():
                if gt_idx in matched_gt:
                    continue

                gt_center = (gt["x"], gt["y"])

                # Matching radius based on nodule size
                radius = max(gt["w"], gt["h"]) / 2.0

                dist = euclidean(pred_center, gt_center)

                if dist <= radius and dist < best_dist:
                    found_match = True
                    best_gt_idx = gt_idx
                    best_dist = dist

            if found_match:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
            
            print(f"Threshold: {thr:.4f}, TP: {tp}, FP: {fp}, Matched GT: {len(matched_gt)}/{num_gt}")

        sensitivity = tp / num_gt
        fp_per_scan = fp / num_scans

        results.append({
            "threshold": thr,
            "TP": tp,
            "FP": fp,
            "sensitivity": sensitivity,
            "fp_per_scan": fp_per_scan
        })

    froc_df = pd.DataFrame(results)

    sensitivities_at_fp = {}

    for rate in fp_rates:
        valid = froc_df[froc_df["fp_per_scan"] <= rate]

        if len(valid) > 0:
            sensitivities_at_fp[rate] = valid["sensitivity"].max()
        else:
            sensitivities_at_fp[rate] = 0.0

    froc_score = np.mean(list(sensitivities_at_fp.values()))

    return froc_df, sensitivities_at_fp, froc_score


if __name__ == "__main__":
    pred_df = pd.read_csv(PRED_PATH)
    gt_df = pd.read_csv(GT_PATH)

    froc_df, sens_at_fp, froc_score = compute_froc(pred_df, gt_df)

    print("Sensitivity at predefined FP/scan rates:")
    for fp_rate, sens in sens_at_fp.items():
        print(f"{fp_rate} FP/scan: {sens:.4f}")

    print(f"\nMean FROC score: {froc_score:.4f}")

    plt.figure(figsize=(7, 5))
    plt.plot(froc_df["fp_per_scan"], froc_df["sensitivity"], marker="o")
    plt.xscale("log")
    plt.xlabel("False positives per scan")
    plt.ylabel("Sensitivity")
    plt.title("FROC Curve")
    plt.grid(True)
    plt.show()