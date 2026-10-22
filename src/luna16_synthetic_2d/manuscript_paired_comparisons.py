"""Recompute paired manuscript comparisons from pooled out-of-fold predictions.

Usage: python -m src.luna16_synthetic_2d.manuscript_paired_comparisons
Writes outputs/manuscript_paired_comparisons.csv. DeLong covariance is computed
from the positive- and negative-case placement values of both paired scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.luna16_synthetic_2d.export_backbone_summary import parse_prediction_files


ROOT = Path("outputs")
RBF_ROOT = ROOT / "luna16_synthetic_2d_cpmnetv2_bf16_top4_minprob0.50_rbf_v100"
SHEPARD_ROOT = ROOT / "luna16_synthetic_2d_cpmnetv2_bf16_top4_minprob0.50_shepard_v100"

# A source is either a pooled prediction CSV or a directory of fold predictions.
COMPARATORS: list[tuple[str, Path, str | None]] = [
    ("Adaptive Shepard", SHEPARD_ROOT / "all_test_predictions.csv", "efficientnet_v2_s"),
    ("MIP axial", ROOT / "luna16_2d_baseline_mip_axial_256x384_efficientnet_v2_s", None),
    ("MIP tri-view", ROOT / "luna16_2d_baseline_mip_triview_256x384_efficientnet_v2_s", None),
    ("Central axial slice", ROOT / "luna16_2d_baseline_central_slice_central_axial_256x384_efficientnet_v2_s", None),
    ("Soft slice attention", ROOT / "luna16_slice_attention_2p5d_effnetb0_half_frozen_v100_bs2" / "all_test_predictions_efficientnet_b0.csv", None),
    ("Detector top-score axial slice", ROOT / "luna16_2d_baseline_detector_top_axial_256x384_efficientnet_v2_s" / "all_test_predictions.csv", None),
    ("Fixed-control RBF", ROOT / "luna16_synthetic_2d_cpmnetv2_bf16_top4_minprob0.50_rbf_fixed_control", None),
    ("Random-control RBF", ROOT / "luna16_synthetic_2d_cpmnetv2_bf16_top4_minprob0.50_rbf_random_control", None),
    ("3D ResNet18", ROOT / "luna16_volume_3d_resnet18_fitpad_224x288x288_b4_acc2_ep100_noes_wandb", None),
    ("Rad-JEPA-3D resize-32", ROOT / "luna16_radjepa_3d/probes_full100/all_test_predictions_resize32.csv", None),
    ("Rad-JEPA-3D sliding mean", ROOT / "luna16_radjepa_3d/probes_full100/all_test_predictions_sliding_mean.csv", None),
    ("Rad-JEPA-3D sliding max", ROOT / "luna16_radjepa_3d/probes_full100/all_test_predictions_sliding_max.csv", None),
    ("Rad-JEPA-3D sliding mean+max", ROOT / "luna16_radjepa_3d/probes_full100/all_test_predictions_sliding_mean_max.csv", None),
    ("3DINO-ViT partial FT, mean+max", ROOT / "luna16_3dino_3d_restart_20260918/all_test_predictions_3dino_partial_ft.csv", None),
    ("3DINO-ViT frozen whole volume", ROOT / "luna16_3dino_frozen_probe/whole/oof_test_predictions.csv", None),
    ("3DINO-ViT frozen detector crops", ROOT / "luna16_3dino_frozen_probe/detections/oof_test_predictions.csv", None),
]


def load_predictions(source: Path, backbone: str | None = None) -> pd.DataFrame:
    data = pd.read_csv(source) if source.is_file() else parse_prediction_files(source)
    if backbone is not None:
        data = data[data["backbone"] == backbone]
    if "sample_id" not in data and "seriesuid" in data:
        data = data.rename(columns={"seriesuid": "sample_id"})
    data = data[["sample_id", "fold", "label", "prediction", "score"]].copy()
    data["sample_id"] = data["sample_id"].astype(str)
    if len(data) != 796 or data["sample_id"].nunique() != 796 or data["fold"].nunique() != 10:
        raise ValueError(f"Expected 796 unique out-of-fold predictions from 10 folds: {source}")
    return data


def paired_delong(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    scores = np.stack([first, second])
    positives = scores[:, labels == 1]
    negatives = scores[:, labels == 0]
    placements = (positives[:, :, None] > negatives[:, None, :]).astype(float)
    placements += 0.5 * (positives[:, :, None] == negatives[:, None, :])
    v10 = placements.mean(axis=2)
    v01 = placements.mean(axis=1)
    aucs = placements.mean(axis=(1, 2))
    covariance = np.cov(v10) / positives.shape[1] + np.cov(v01) / negatives.shape[1]
    delta = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    if variance <= 0:
        raise ValueError("Nonpositive DeLong variance for paired AUC difference")
    p_value = float(2 * norm.sf(abs(delta) / np.sqrt(variance)))
    return delta, p_value


def bh_fdr(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranks = np.arange(1, len(values) + 1)
    adjusted_sorted = np.minimum.accumulate((values[order] * len(values) / ranks)[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.clip(adjusted_sorted, 0, 1)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON manifest with reference and comparators; paths relative to the manifest.")
    parser.add_argument("--output", type=Path, default=ROOT / "manuscript_paired_comparisons.csv")
    args = parser.parse_args()
    reference_path, reference_backbone = RBF_ROOT / "all_test_predictions.csv", "efficientnet_v2_s"
    comparators = COMPARATORS
    if args.config:
        manifest = json.loads(args.config.read_text())
        def source_path(item):
            return args.config.parent / item["path"]
        reference_path = source_path(manifest["reference"])
        reference_backbone = manifest["reference"].get("backbone")
        comparators = [(item["name"], source_path(item), item.get("backbone")) for item in manifest["comparators"]]
    names = [item[0] for item in comparators]
    if not names or len(names) != len(set(names)):
        raise ValueError("Provide at least one comparator with unique names")
    reference = load_predictions(reference_path, reference_backbone)
    rows = []
    for name, source, backbone in comparators:
        comparator = load_predictions(source, backbone)
        paired = reference.merge(comparator, on="sample_id", suffixes=("_rbf", "_other"), validate="one_to_one")
        if len(paired) != len(reference):
            raise ValueError(f"Patient cohort mismatch for {name}")
        if not (paired["label_rbf"] == paired["label_other"]).all():
            raise ValueError(f"Label mismatch for {name}")
        if not (paired["fold_rbf"] == paired["fold_other"]).all():
            raise ValueError(f"Fold mismatch for {name}")

        labels = paired["label_rbf"].to_numpy(dtype=int)
        delta, p_delong = paired_delong(
            labels, paired["score_rbf"].to_numpy(dtype=float), paired["score_other"].to_numpy(dtype=float)
        )
        rbf_correct = paired["prediction_rbf"].to_numpy(dtype=int) == labels
        other_correct = paired["prediction_other"].to_numpy(dtype=int) == labels
        rbf_only = int((rbf_correct & ~other_correct).sum())
        other_only = int((~rbf_correct & other_correct).sum())
        discordant = rbf_only + other_only
        p_mcnemar = float(binomtest(min(rbf_only, other_only), discordant, 0.5).pvalue) if discordant else 1.0
        rows.append(
            {
                "comparator": name,
                "n": len(paired),
                "delta_auc": delta,
                "p_delong": p_delong,
                "rbf_correct_only": rbf_only,
                "comparator_correct_only": other_only,
                "p_mcnemar": p_mcnemar,
            }
        )

    results = pd.DataFrame(rows)
    results["q_delong"] = bh_fdr(results["p_delong"])
    results["q_mcnemar"] = bh_fdr(results["p_mcnemar"])
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
