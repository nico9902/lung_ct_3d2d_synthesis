"""Quantify Grad-CAM energy inside reference-nodule regions on the full test set.

For every fold, the EfficientNetV2-S model of the proposed method is applied to
all test scans of that fold. Grad-CAM is computed (un-normalized ReLU map,
upsampled to the classifier input) for the predicted class and for the
malignant class, and the fraction of Grad-CAM energy falling inside the
reference-nodule region is reported together with the area fraction of that
region (the expected energy fraction of a spatially uniform map).

Two nodule regions are evaluated:
- projection: axial projection of the nodule mask (as drawn in the paper figure);
- sampled: nodule voxels actually traversed by the RBF sampling surface,
  i.e. pixels (y, x) with nodule_mask[S(y, x), y, x] = 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy import stats

from .dataset import SyntheticLuna16Dataset
from .explain_gradcam import (
    build_explain_transforms,
    class_objective,
    get_target_layer,
    load_mask_zyx,
    load_model,
    load_run_config,
    probabilities_from_logits,
    resize_mask,
)

CLASSES = ["benign", "malignant"]
MALIGNANT_RATING = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("outputs/luna16_synthetic_2d_cpmnetv2_bf16_top4_minprob0.50_rbf_v100"))
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument(
        "--synthetic-images-dir",
        type=Path,
        default=Path("data/synthetic_2d/luna16_saliency_synthetic_detector_cpmnetv2_bf16_top4_minprob0.50_rbf"),
    )
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/processed/luna16_labels.csv"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/processed/cv_splits"))
    parser.add_argument("--image-suffix", default="_tps_top5.npy")
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 384])
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/luna16_synthetic_2d_gradcam_nodule_energy"))
    parser.add_argument("--save-cams", action="store_true")
    return parser.parse_args()


def raw_gradcam(model, tensor: torch.Tensor, target_class: int, target_layer) -> tuple[np.ndarray, np.ndarray]:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    forward_handle = target_layer.register_forward_hook(lambda _m, _i, out: activations.append(out))
    backward_handle = target_layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        probabilities = probabilities_from_logits(logits).squeeze(0).detach().cpu().numpy()
        class_objective(logits, target_class).backward()
        weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[-1]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
    finally:
        forward_handle.remove()
        backward_handle.remove()
    return np.nan_to_num(cam.squeeze().detach().cpu().numpy()).astype(np.float64), probabilities


def component_ratings(labeled: np.ndarray, count: int, centers: pd.DataFrame, ratings: list[float]) -> list[float] | None:
    """Assign each mask component the rating of the nearest reference-nodule center.

    Ratings follow the order of the LUNA16 annotations, as do the processed
    centers in luna16_labels.csv; components are matched to centers by distance.
    """
    if len(centers) != len(ratings) or count == 0:
        return None
    centroids = np.asarray(ndi.center_of_mass(labeled > 0, labeled, range(1, count + 1)))
    points = centers[["z", "y", "x"]].to_numpy(dtype=float)
    assigned = []
    for centroid in centroids:
        distances = np.linalg.norm(points - centroid, axis=1)
        assigned.append(ratings[int(np.argmin(distances))])
    return assigned


def nodule_regions(
    row: pd.Series,
    processed_root: Path,
    centers: pd.DataFrame,
    surface: np.ndarray,
    shape: tuple[int, int],
) -> dict[str, np.ndarray] | None:
    if pd.isna(row["nodule_mask_path"]):
        return None
    mask_path = processed_root / str(row["nodule_mask_path"])
    nodule_mask = load_mask_zyx(mask_path) > 0
    if not nodule_mask.any():
        return None
    lung_path = mask_path.with_name(mask_path.name.replace("_nodule_mask", "_lung_mask"))
    if lung_path.exists():
        lung_mask = load_mask_zyx(lung_path) > 0
        if lung_mask.shape == nodule_mask.shape:
            nodule_mask &= lung_mask
    labeled, count = ndi.label(nodule_mask)
    ratings = component_ratings(labeled, count, centers, json.loads(str(row["nodule_annotation_mean_malignancies"])))

    if surface.shape != nodule_mask.shape[1:]:
        raise ValueError(f"Surface {surface.shape} does not match mask {nodule_mask.shape} for {row['seriesuid']}")
    rows, cols = np.indices(surface.shape)
    sampled_labels = labeled[np.clip(surface, 0, labeled.shape[0] - 1), rows, cols]
    projected_labels = labeled.max(axis=0)

    regions = {}
    for region, label_map in (("projection", projected_labels), ("sampled", sampled_labels)):
        all_ids = list(range(1, count + 1))
        regions[f"{region}_all"] = resize_mask(np.isin(label_map, all_ids), shape)
        if ratings is None:
            regions[f"{region}_malignant"] = None
        else:
            malignant_ids = [i for i in all_ids if ratings[i - 1] > MALIGNANT_RATING]
            regions[f"{region}_malignant"] = resize_mask(np.isin(label_map, malignant_ids), shape)
    regions["n_nodules"] = count
    regions["n_nodules_sampled"] = len(np.unique(sampled_labels[sampled_labels > 0]))
    regions["n_malignant"] = -1 if ratings is None else sum(r > MALIGNANT_RATING for r in ratings)
    return regions


def region_metrics(cam: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    total = cam.sum()
    if not mask.any() or total <= 0:
        return {"energy_fraction": np.nan, "area_fraction": float(mask.mean()), "enrichment": np.nan, "pointing_hit": np.nan}
    energy = float(cam[mask].sum() / total)
    area = float(mask.mean())
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return {"energy_fraction": energy, "area_fraction": area, "enrichment": energy / area, "pointing_hit": float(mask[peak])}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    labels = pd.read_csv(args.labels_csv, dtype={"seriesuid": str})
    centers_by_scan = {uid: group for uid, group in labels.groupby("seriesuid", sort=False)}
    empty_centers = labels.iloc[0:0]

    for fold in args.folds:
        run_dir = args.run_root / f"fold_{fold}" / args.backbone
        checkpoints = sorted(p for p in (run_dir / "checkpoints").glob("*.ckpt") if not p.name.startswith("last"))
        checkpoint = checkpoints[0]
        split_csv = args.splits_dir / f"luna16_classification_fold{fold}.csv"
        metadata = pd.read_csv(split_csv, dtype={"seriesuid": str}).set_index("seriesuid")
        model = load_model(checkpoint, args.backbone, CLASSES, load_run_config(run_dir, checkpoint), device)
        target_layer = get_target_layer(model, None)
        dataset = SyntheticLuna16Dataset(
            synthetic_images_dir=args.synthetic_images_dir,
            split_csv=split_csv,
            fold=fold,
            split="test",
            classes=CLASSES,
            transform=build_explain_transforms(args.image_size),
            image_suffix=args.image_suffix,
        )
        print(f"Fold {fold}: {len(dataset)} test scans, checkpoint {checkpoint.name}", flush=True)

        for index in range(len(dataset)):
            tensor, label, sample_id = dataset[index]
            tensor = tensor.unsqueeze(0).to(device)
            row = metadata.loc[sample_id].copy()
            row["seriesuid"] = sample_id
            centers = centers_by_scan.get(sample_id, empty_centers)
            surface = np.load(args.synthetic_images_dir / sample_id / f"surface_grid_int_{sample_id}.npy").astype(int)
            regions = nodule_regions(row, args.processed_root, centers, surface, tuple(args.image_size))

            cams = {}
            cams["malignant"], probabilities = raw_gradcam(model, tensor, 1, target_layer)
            predicted = int(np.argmax(probabilities))
            cams["predicted"] = cams["malignant"] if predicted == 1 else raw_gradcam(model, tensor, 0, target_layer)[0]

            base = {
                "fold": fold,
                "sample_id": sample_id,
                "label": int(label),
                "prediction": predicted,
                "p_malignant": float(probabilities[1]) if probabilities.size > 1 else float(probabilities[0]),
                "correct": int(predicted == int(label)),
                "n_nodules": 0 if regions is None else regions["n_nodules"],
                "n_nodules_sampled": 0 if regions is None else regions["n_nodules_sampled"],
                "n_malignant": 0 if regions is None else regions["n_malignant"],
            }
            for target, cam in cams.items():
                if args.save_cams:
                    cam_dir = args.output_dir / "cams" / f"fold_{fold}"
                    cam_dir.mkdir(parents=True, exist_ok=True)
                    np.save(cam_dir / f"{sample_id}_{target}.npy", cam.astype(np.float16))
                for region in ("projection_all", "projection_malignant", "sampled_all", "sampled_malignant"):
                    mask = None if regions is None else regions[region]
                    if mask is None:
                        mask = np.zeros(cam.shape, dtype=bool)
                    record = dict(base, cam_target=target, region=region)
                    record.update(region_metrics(cam, mask))
                    records.append(record)

        del model
        torch.cuda.empty_cache()

    per_scan = pd.DataFrame(records)
    per_scan.to_csv(args.output_dir / "per_scan_metrics.csv", index=False)
    summary = summarize(per_scan)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary.to_dict(orient="records"), handle, indent=2)
    print(summary.to_string(index=False))


def summarize(per_scan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strata = {
        "all_with_region": lambda d: d,
        "positive": lambda d: d[d["label"] == 1],
        "negative": lambda d: d[d["label"] == 0],
        "positive_correct": lambda d: d[(d["label"] == 1) & (d["correct"] == 1)],
        "negative_correct": lambda d: d[(d["label"] == 0) & (d["correct"] == 1)],
    }
    for (target, region), group in per_scan.groupby(["cam_target", "region"]):
        valid = group.dropna(subset=["energy_fraction"])
        for stratum, select in strata.items():
            data = select(valid)
            if data.empty:
                continue
            wilcoxon_p = stats.wilcoxon(data["energy_fraction"], data["area_fraction"], alternative="greater").pvalue if len(data) > 1 else np.nan
            rows.append(
                {
                    "cam_target": target,
                    "region": region,
                    "stratum": stratum,
                    "n_scans": len(data),
                    "energy_fraction_mean": data["energy_fraction"].mean(),
                    "energy_fraction_median": data["energy_fraction"].median(),
                    "energy_fraction_q1": data["energy_fraction"].quantile(0.25),
                    "energy_fraction_q3": data["energy_fraction"].quantile(0.75),
                    "area_fraction_mean": data["area_fraction"].mean(),
                    "area_fraction_median": data["area_fraction"].median(),
                    "enrichment_median": data["enrichment"].median(),
                    "pointing_hit_rate": data["pointing_hit"].mean(),
                    "wilcoxon_energy_gt_area_p": wilcoxon_p,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
