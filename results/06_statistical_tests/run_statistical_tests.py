from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
REMOTE = ROOT / "predictions" / "remote" / "outputs"
N_BOOTSTRAP = 5000
RANDOM_SEED = 20260819


METHODS = {
    "adaptive_rbf_effnetv2s": {
        "label": "Adaptive RBF, detector-guided, top4/thr0.50, EfficientNetV2-S",
        "kind": "all_test",
        "path": REMOTE / "luna16_synthetic_2d_cpmnetv2_bf16_top4_minprob0.50_rbf_v100" / "all_test_predictions.csv",
        "backbone": "efficientnet_v2_s",
    },
    "adaptive_shepard_effnetv2s": {
        "label": "Adaptive Shepard, detector-guided, top4/thr0.50, EfficientNetV2-S",
        "kind": "all_test",
        "path": REMOTE / "luna16_synthetic_2d_cpmnetv2_bf16_top4_minprob0.50_shepard_v100" / "all_test_predictions.csv",
        "backbone": "efficientnet_v2_s",
    },
    "mip_axial_effnetv2s": {
        "label": "MIP axial, EfficientNetV2-S",
        "kind": "folds",
        "path": REMOTE / "luna16_2d_baseline_mip_axial_256x384_efficientnet_v2_s",
        "pattern": "fold_*/efficientnet_v2_s/test_predictions.csv",
    },
    "mip_triview_effnetv2s": {
        "label": "MIP tri-view, EfficientNetV2-S",
        "kind": "folds",
        "path": REMOTE / "luna16_2d_baseline_mip_triview_256x384_efficientnet_v2_s",
        "pattern": "fold_*/efficientnet_v2_s/test_predictions.csv",
    },
    "central_slice_effnetv2s": {
        "label": "Central axial slice, EfficientNetV2-S",
        "kind": "folds",
        "path": REMOTE / "luna16_2d_baseline_central_slice_central_axial_256x384_efficientnet_v2_s",
        "pattern": "fold_*/efficientnet_v2_s/test_predictions.csv",
    },
    "crop_mil_attention_effnetv2s": {
        "label": "Detector crop-MIL attention, EfficientNetV2-S",
        "kind": "folds",
        "path": REMOTE / "luna16_detection_mil_cpmnetv2_top4_minprob0.50_effnetv2s",
        "pattern": "fold_*/attention/test_predictions.csv",
    },
    "fixed_control_rbf_effnetv2s": {
        "label": "Fixed-control RBF, EfficientNetV2-S",
        "kind": "folds",
        "path": REMOTE / "luna16_synthetic_2d_cpmnetv2_bf16_top4_minprob0.50_rbf_fixed_control",
        "pattern": "fold_*/efficientnet_v2_s/test_predictions.csv",
    },
    "random_control_rbf_effnetv2s": {
        "label": "Random-control RBF, EfficientNetV2-S",
        "kind": "folds",
        "path": REMOTE / "luna16_synthetic_2d_cpmnetv2_bf16_top4_minprob0.50_rbf_random_control",
        "pattern": "fold_*/efficientnet_v2_s/test_predictions.csv",
    },
    "resnet18_3d_fitpad": {
        "label": "3D ResNet18, fit-pad 224x288x288",
        "kind": "folds",
        "path": REMOTE / "luna16_volume_3d_resnet18_fitpad_224x288x288_b4_acc2_ep100_noes_wandb",
        "pattern": "fold_*/resnet18_3d/test_predictions.csv",
    },
}


COMPARISONS = [
    ("adaptive_rbf_effnetv2s", "adaptive_shepard_effnetv2s"),
    ("adaptive_rbf_effnetv2s", "mip_axial_effnetv2s"),
    ("adaptive_rbf_effnetv2s", "mip_triview_effnetv2s"),
    ("adaptive_rbf_effnetv2s", "central_slice_effnetv2s"),
    ("adaptive_rbf_effnetv2s", "crop_mil_attention_effnetv2s"),
    ("adaptive_rbf_effnetv2s", "fixed_control_rbf_effnetv2s"),
    ("adaptive_rbf_effnetv2s", "random_control_rbf_effnetv2s"),
    ("adaptive_rbf_effnetv2s", "resnet18_3d_fitpad"),
]


def load_predictions(spec: dict) -> pd.DataFrame:
    if spec["kind"] == "all_test":
        df = pd.read_csv(spec["path"])
        if "backbone" in df.columns and spec.get("backbone"):
            df = df[df["backbone"] == spec["backbone"]].copy()
    else:
        files = sorted(spec["path"].glob(spec["pattern"]))
        if not files:
            raise FileNotFoundError(f"No files for {spec['path']} / {spec['pattern']}")
        df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    columns = ["sample_id", "label", "prediction", "score"]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing {missing} in {spec}")
    df = df[columns].copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["label"] = df["label"].astype(int)
    df["prediction"] = df["prediction"].astype(int)
    df["score"] = df["score"].astype(float)
    df = df.sort_values("sample_id").drop_duplicates("sample_id", keep="last").reset_index(drop=True)
    return df


def metrics_for(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "samples": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int((y == 0).sum()),
        "auc": float(roc_auc_score(y, score)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    ranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x), dtype=float)
    out[order] = ranks
    return out


def fast_delong(predictions_sorted: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    n_neg = predictions_sorted.shape[1] - n_pos
    tx = np.array([compute_midrank(row[:n_pos]) for row in predictions_sorted])
    ty = np.array([compute_midrank(row[n_pos:]) for row in predictions_sorted])
    tz = np.array([compute_midrank(row) for row in predictions_sorted])
    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / n_pos + sy / n_neg
    return aucs, np.atleast_2d(delong_cov)


def delong_pvalue(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> tuple[float, float, float]:
    order = np.argsort(-y)
    n_pos = int(y.sum())
    predictions = np.vstack([score_a, score_b])[:, order]
    aucs, cov = fast_delong(predictions, n_pos)
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ cov @ contrast.T).item())
    delta = float(aucs[0] - aucs[1])
    if variance <= 0:
        return delta, float("nan"), float("nan")
    z = abs(delta) / np.sqrt(variance)
    p = 2.0 * stats.norm.sf(z)
    return delta, float(z), float(p)


def mcnemar_exact(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    correct_a = pred_a == y
    correct_b = pred_b == y
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
    return {"a_correct_b_wrong": b, "a_wrong_b_correct": c, "mcnemar_p": p}


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return q.tolist()
    order = valid_idx[np.argsort(p[valid])]
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0, 1)
    return q.tolist()


def bootstrap_ci(y: np.ndarray, score: np.ndarray, pred: np.ndarray, rng: np.random.Generator) -> dict:
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    values = {metric: [] for metric in ["auc", "mcc", "accuracy", "f1", "precision", "recall"]}
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        m = metrics_for(y[idx], score[idx], pred[idx])
        for metric in values:
            values[metric].append(m[metric])
    rows = {}
    for metric, vals in values.items():
        arr = np.asarray(vals)
        rows[f"{metric}_ci_low"] = float(np.percentile(arr, 2.5))
        rows[f"{metric}_ci_high"] = float(np.percentile(arr, 97.5))
    return rows


def align_pair(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    merged = df_a.merge(df_b, on="sample_id", suffixes=("_a", "_b"), how="inner")
    if len(merged) != len(df_a) or len(merged) != len(df_b):
        raise ValueError(f"Pair alignment changed sample size: {len(df_a)} vs {len(df_b)} -> {len(merged)}")
    if not np.array_equal(merged["label_a"].to_numpy(), merged["label_b"].to_numpy()):
        raise ValueError("Labels differ after alignment")
    return merged


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    predictions = {name: load_predictions(spec) for name, spec in METHODS.items()}

    pooled_rows = []
    ci_rows = []
    for name, df in predictions.items():
        y = df["label"].to_numpy()
        score = df["score"].to_numpy()
        pred = df["prediction"].to_numpy()
        row = {"method": name, "label": METHODS[name]["label"], **metrics_for(y, score, pred)}
        pooled_rows.append(row)
        ci_rows.append({"method": name, "label": METHODS[name]["label"], **bootstrap_ci(y, score, pred, rng)})

    pooled = pd.DataFrame(pooled_rows).sort_values("auc", ascending=False)
    ci = pd.DataFrame(ci_rows)
    pooled.to_csv(ROOT / "pooled_metrics_for_stat_tests.csv", index=False)
    ci.to_csv(ROOT / "bootstrap_95ci.csv", index=False)

    comparison_rows = []
    for a, b in COMPARISONS:
        pair = align_pair(predictions[a], predictions[b])
        y = pair["label_a"].to_numpy()
        score_a = pair["score_a"].to_numpy()
        score_b = pair["score_b"].to_numpy()
        pred_a = pair["prediction_a"].to_numpy()
        pred_b = pair["prediction_b"].to_numpy()
        delta_auc, z, delong_p = delong_pvalue(y, score_a, score_b)
        mc = mcnemar_exact(y, pred_a, pred_b)
        metrics_a = metrics_for(y, score_a, pred_a)
        metrics_b = metrics_for(y, score_b, pred_b)
        comparison_rows.append(
            {
                "method_a": a,
                "method_b": b,
                "label_a": METHODS[a]["label"],
                "label_b": METHODS[b]["label"],
                "samples": int(len(y)),
                "auc_a": metrics_a["auc"],
                "auc_b": metrics_b["auc"],
                "delta_auc": delta_auc,
                "delong_z": z,
                "delong_p": delong_p,
                "mcc_a": metrics_a["mcc"],
                "mcc_b": metrics_b["mcc"],
                "delta_mcc": metrics_a["mcc"] - metrics_b["mcc"],
                "f1_a": metrics_a["f1"],
                "f1_b": metrics_b["f1"],
                "delta_f1": metrics_a["f1"] - metrics_b["f1"],
                **mc,
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["delong_p_fdr_bh"] = benjamini_hochberg(comparisons["delong_p"].tolist())
    comparisons["mcnemar_p_fdr_bh"] = benjamini_hochberg(comparisons["mcnemar_p"].tolist())
    comparisons.to_csv(ROOT / "paired_comparison_tests.csv", index=False)

    summary = {
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap": "patient-level stratified bootstrap over pooled 10-fold test predictions",
        "auc_test": "paired DeLong test",
        "binary_test": "exact McNemar/binomial test on paired 0.5-threshold predictions",
        "multiple_testing": "Benjamini-Hochberg FDR across planned pairwise comparisons",
        "methods": {name: spec["label"] for name, spec in METHODS.items()},
        "comparisons": COMPARISONS,
    }
    with (ROOT / "statistical_test_protocol.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
