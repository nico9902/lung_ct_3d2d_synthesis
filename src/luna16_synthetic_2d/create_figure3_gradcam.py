#!/usr/bin/env python3
"""Create the split benign/malignant Grad-CAM panel used as manuscript Figure 3."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi


BENIGN_CASES = [
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.832260670372728970918746541371",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.277445975068759205899107114231",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.323302986710576400812869264321",
]
MALIGNANT_CASES = [
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.334517907433161353885866806005",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.534006575256943390479252771547",
]

GREEN = "#00B050"
RED = "#E53935"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(
            "outputs/luna16_synthetic_2d_gradcam/"
            "luna16_synthetic_2d_top4_minprob0.5_rbf_efficientnet_v2_s/fold_0"
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/processed/cv_splits/luna16_classification_fold0.csv"),
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("docs/luna16_synthetic_2d_gradcam_all_predicted/gradcam_manifest.csv"),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/ssd2/domenico/datasets/LUNA16_preprocessed"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manuscript/figures/gradcam_summary.png"),
    )
    return parser.parse_args()


def parse_scores(value: object) -> list[float]:
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    return [float(score) for score in parsed]


def load_mask_zyx(path: Path) -> np.ndarray:
    array = nib.load(str(path)).get_fdata().astype(np.float32)
    return array.transpose(2, 1, 0) > 0


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def locate_asset(folder: Path, sample_id: str, suffix: str) -> Path:
    matches = sorted(folder.glob(f"*_{sample_id}_*_{suffix}.png"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {suffix} asset for {sample_id} in {folder}, found {len(matches)}"
        )
    return matches[0]


def projected_components(
    mask_path: Path,
    lung_mask_path: Path,
    scores: list[float],
    shape: tuple[int, int],
):
    labeled, count = ndi.label(load_mask_zyx(mask_path))
    if count != len(scores):
        raise ValueError(
            f"Mask/score mismatch for {mask_path}: {count} components and {len(scores)} scores"
        )
    lung_mask = load_mask_zyx(lung_mask_path)
    components = []
    for component_id in range(1, count + 1):
        component = (labeled == component_id) & lung_mask
        if not component.any():
            raise ValueError(f"Nodule component {component_id} lies outside the lung mask: {mask_path}")
        components.append(
            (resize_mask(component.max(axis=0), shape), scores[component_id - 1])
        )
    return components


def visible_lung_coverages(image: np.ndarray, components) -> list[float]:
    visible = image.astype(np.float32).mean(axis=2) / 255.0 > 0.03
    visible = ndi.binary_dilation(visible, iterations=3)
    return [
        float((mask & visible).sum() / max(1, mask.sum()))
        for mask, _score in components
    ]


def add_contours(ax: plt.Axes, components) -> None:
    for mask, score in components:
        color = RED if score > 3.0 else GREEN
        ax.contour(
            mask.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=1.15,
            antialiased=True,
        )


def prepare_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    metadata = pd.read_csv(args.metadata_csv)
    predictions = pd.read_csv(args.predictions_csv)
    predictions = predictions[
        (predictions["experiment"] == "luna16_synthetic_2d_top4_minprob0.5_rbf")
        & (predictions["backbone"] == "efficientnet_v2_s")
        & (predictions["fold"] == 0)
    ]

    records = []
    for group, sample_ids in (("benign", BENIGN_CASES), ("malignant", MALIGNANT_CASES)):
        for row_number, sample_id in enumerate(sample_ids, start=1):
            row = metadata[metadata["seriesuid"].astype(str) == sample_id]
            pred = predictions[predictions["sample_id"].astype(str) == sample_id]
            if len(row) != 1 or len(pred) != 1:
                raise ValueError(f"Could not uniquely resolve metadata/prediction for {sample_id}")
            row = row.iloc[0]
            pred = pred.iloc[0]
            image_path = locate_asset(args.assets_dir / "images", sample_id, "image")
            overlay_path = locate_asset(args.assets_dir / "overlays", sample_id, "overlay")
            image = np.asarray(Image.open(image_path).convert("RGB"))
            overlay = np.asarray(Image.open(overlay_path).convert("RGB"))
            relative_mask_path = Path(str(row["nodule_mask_path"]))
            mask_path = args.processed_root / relative_mask_path
            lung_mask_path = mask_path.with_name(mask_path.name.replace("_nodule_mask", "_lung_mask"))
            scores = parse_scores(row["nodule_annotation_mean_malignancies"])
            components = projected_components(mask_path, lung_mask_path, scores, image.shape[:2])
            coverages = visible_lung_coverages(image, components)
            if min(coverages) < 0.95:
                raise ValueError(
                    f"Projected nodule outside visible lung for {sample_id}: coverages={coverages}"
                )
            records.append(
                {
                    "group": group,
                    "case": f"{'B' if group == 'benign' else 'M'}{row_number}",
                    "sample_id": sample_id,
                    "image": image,
                    "overlay": overlay,
                    "components": components,
                    "scores": scores,
                    "p_malignant": (
                        1.0 - float(pred["true_class_score"])
                        if group == "benign"
                        else float(pred["true_class_score"])
                    ),
                    "prediction": str(pred["prediction_name"]),
                    "image_path": str(image_path),
                    "overlay_path": str(overlay_path),
                    "mask_path": str(mask_path),
                    "visible_lung_coverages": coverages,
                }
            )
    return records


def render(cases: list[dict[str, object]], output: Path) -> None:
    by_group = {
        group: [case for case in cases if case["group"] == group]
        for group in ("benign", "malignant")
    }
    fig, axes = plt.subplots(3, 4, figsize=(12.4, 6.7), dpi=300)
    plt.subplots_adjust(left=0.018, right=0.982, bottom=0.105, top=0.895, wspace=0.025, hspace=0.075)

    for row_number in range(3):
        row_cases = (by_group["benign"][row_number], by_group["malignant"][row_number])
        for group_number, case in enumerate(row_cases):
            for local_col, key in enumerate(("image", "overlay")):
                col = group_number * 2 + local_col
                ax = axes[row_number, col]
                ax.imshow(case[key])
                add_contours(ax, case["components"])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if local_col == 0:
                    ax.text(
                        0.018,
                        0.965,
                        f"{case['case']}  P(malignant)={case['p_malignant']:.3f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=7.5,
                        color="white",
                        bbox={"boxstyle": "round,pad=0.22", "facecolor": "black", "alpha": 0.72, "edgecolor": "none"},
                    )

    column_titles = ("Synthetic image", "Grad-CAM", "Synthetic image", "Grad-CAM")
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=9.5, pad=5)

    fig.text(0.255, 0.955, "Benign patients", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.text(0.745, 0.955, "Malignant patients", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.add_artist(Line2D([0.5, 0.5], [0.105, 0.965], transform=fig.transFigure, color="#777777", linewidth=0.8))
    legend_handles = [
        Line2D([0], [0], color=GREEN, lw=2.0, label="Non-malignant / indeterminate nodule (mean rating ≤ 3)"),
        Line2D([0], [0], color=RED, lw=2.0, label="Malignant nodule (mean rating > 3)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=2.4,
        columnspacing=2.0,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), dpi=300, facecolor="white")
    plt.close(fig)


def write_manifest(cases: list[dict[str, object]], output: Path) -> None:
    rows = []
    for case in cases:
        rows.append(
            {
                "case": case["case"],
                "patient_class": case["group"],
                "sample_id": case["sample_id"],
                "prediction": case["prediction"],
                "p_malignant": case["p_malignant"],
                "nodule_mean_malignancies": case["scores"],
                "image_path": case["image_path"],
                "overlay_path": case["overlay_path"],
                "mask_path": case["mask_path"],
                "visible_lung_coverages": case["visible_lung_coverages"],
            }
        )
    pd.DataFrame(rows).to_csv(output.with_name("figure3_cases.csv"), index=False)


def main() -> None:
    args = parse_args()
    cases = prepare_cases(args)
    render(cases, args.output)
    write_manifest(cases, args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
