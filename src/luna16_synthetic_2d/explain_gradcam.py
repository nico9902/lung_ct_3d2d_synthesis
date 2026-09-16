from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.luna16_synthetic_2d.dataset import SyntheticLuna16Dataset
else:
    from .dataset import SyntheticLuna16Dataset


SUPPORTED_BACKBONES = (
    "vgg16",
    "vgg19",
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet121",
    "densenet169",
    "densenet201",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_v2_s",
)


CLASS_NAME_TO_INDEX = {
    "benign": 0,
    "malignant": 1,
}


@dataclass
class GradCamOutput:
    display_image: np.ndarray
    heatmap: np.ndarray
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_class: int
    target_class: int


@dataclass
class GroundTruthOverlay:
    image: np.ndarray
    available: bool
    description: str
    mask: np.ndarray | None = None
    clipped_to_lung: bool = False
    removed_outside_lung_voxels: int = 0


class MinMaxScale:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        min_value = tensor.amin()
        max_value = tensor.amax()
        scale = (max_value - min_value).clamp_min(1e-6)
        return (tensor - min_value) / scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Grad-CAM heatmaps and overlay figures for trained LUNA16 "
            "synthetic-2D classifiers."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a Lightning .ckpt file.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional training run directory containing config.json. Defaults to checkpoint parent ancestors.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/luna16_synthetic_2d_gradcam"))
    parser.add_argument("--synthetic-images-dir", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--classes", nargs="+", default=["benign", "malignant"])
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, default=None)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=None,
        help="Resize as H W, or one value for square. Defaults to run config or 256 384.",
    )
    parser.add_argument(
        "--image-suffix",
        default="_tps_top5.npy",
        help="Suffix used by SyntheticLuna16Dataset when resolving flat fold images.",
    )
    parser.add_argument("--sample-ids", nargs="+", default=None, help="Optional explicit seriesuid list.")
    parser.add_argument("--exclude-sample-ids", nargs="+", default=None, help="Optional seriesuid list to remove before selection.")
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument(
        "--selection",
        choices=[
            "first",
            "correct",
            "incorrect",
            "highest_score",
            "lowest_score",
            "balanced_first",
            "balanced_highest_score",
            "balanced_lowest_score",
        ],
        default="first",
        help="How to choose samples after optional --sample-ids filtering.",
    )
    parser.add_argument(
        "--balance-fill-shortfall",
        action="store_true",
        help=(
            "For balanced_* selections, fill missing slots from other classes when one class has too few samples. "
            "By default balanced selections keep equal class counts, so fewer than --max-samples may be exported."
        ),
    )
    parser.add_argument(
        "--target-class",
        choices=["predicted", "true", "benign", "malignant"],
        default="predicted",
        help="Class whose evidence should be highlighted by Grad-CAM.",
    )
    parser.add_argument(
        "--target-layer",
        default=None,
        help="Optional module name for Grad-CAM. By default the last Conv2d layer is used.",
    )
    parser.add_argument("--alpha", type=float, default=0.42, help="Heatmap opacity in overlays.")
    parser.add_argument("--cmap", default="jet", help="Matplotlib colormap used for heatmaps.")
    parser.add_argument(
        "--figure-layout",
        choices=["paper_overlay", "full"],
        default="paper_overlay",
        help="paper_overlay shows only GT nodule and model overlay; full also includes input and raw heatmap.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0, help="Reserved for CLI symmetry; dataset is iterated directly.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/LUNA16_preprocessed"),
        help="Root containing <seriesuid>/<seriesuid>_nodule_mask.nii.gz for GT overlays.",
    )
    parser.add_argument(
        "--gt-overlay",
        choices=["auto", "always", "never"],
        default="auto",
        help="Whether to include the ground-truth nodule mask panel in per-sample figures.",
    )
    parser.add_argument(
        "--clip-gt-to-lung",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip GT nodule masks to the available lung mask before drawing/filtering.",
    )
    parser.add_argument(
        "--require-malignant-gt-hit",
        action="store_true",
        help="Keep malignant examples only when the Grad-CAM hotspot overlaps the ground-truth nodule mask.",
    )
    parser.add_argument(
        "--require-gt-inside-visible-lung",
        action="store_true",
        help="Discard examples whose projected GT nodule mask falls outside the visible lung silhouette.",
    )
    parser.add_argument(
        "--gt-visible-lung-min-coverage",
        type=float,
        default=0.95,
        help="Minimum GT-mask fraction that must lie inside the visible lung silhouette.",
    )
    parser.add_argument(
        "--visible-lung-threshold",
        type=float,
        default=0.03,
        help="Intensity threshold used to estimate the visible lung/body silhouette from the displayed image.",
    )
    parser.add_argument(
        "--gt-hit-cam-percentile",
        type=float,
        default=90.0,
        help="Percentile used to define Grad-CAM hotspot pixels for --require-malignant-gt-hit.",
    )
    parser.add_argument(
        "--gt-hit-min-mask-coverage",
        type=float,
        default=0.05,
        help="Minimum fraction of GT nodule-mask pixels covered by CAM hotspots to count as a hit.",
    )
    parser.add_argument("--no-summary", action="store_true", help="Skip the combined summary mosaic.")
    parser.add_argument("--summary-cols", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    run_config = load_run_config(args.run_dir, args.checkpoint)
    backbone = args.backbone or str(run_config.get("backbone", "resnet18"))
    image_size = normalize_image_size(args.image_size or run_config.get("image_size", [256, 384]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(
        checkpoint_path=args.checkpoint,
        backbone=backbone,
        classes=args.classes,
        run_config=run_config,
        device=device,
    )
    dataset = SyntheticLuna16Dataset(
        synthetic_images_dir=args.synthetic_images_dir,
        split_csv=args.split_csv,
        fold=args.fold,
        split=args.split,
        classes=args.classes,
        transform=build_explain_transforms(image_size),
        image_suffix=args.image_suffix,
    )

    scored_rows = exclude_sample_ids(score_dataset(model, dataset, device), args.exclude_sample_ids)
    candidate_max_samples = len(scored_rows) if args.require_malignant_gt_hit else args.max_samples
    rows = select_rows(
        scored_rows,
        args.sample_ids,
        args.selection,
        candidate_max_samples,
        args.balance_fill_shortfall,
    )
    if rows.empty:
        raise RuntimeError("No samples selected for Grad-CAM export.")

    records: list[dict[str, object]] = []
    panels: list[Image.Image] = []
    for _, row in rows.iterrows():
        index = int(row["dataset_index"])
        sample = dataset.samples[index]
        target_class = resolve_target_class(args.target_class, int(row["label"]), int(row["prediction"]))
        output = compute_gradcam(
            model=model,
            tensor=dataset[index][0].unsqueeze(0).to(device),
            target_class=target_class,
            target_layer_name=args.target_layer,
        )
        gt_overlay = build_ground_truth_overlay(
            sample_id=sample.sample_id,
            processed_dir=args.processed_dir,
            display_image=output.display_image,
            enabled=args.gt_overlay != "never",
            strict=args.gt_overlay == "always",
            clip_to_lung=args.clip_gt_to_lung,
        )
        if args.gt_overlay == "always" and not gt_overlay.available:
            raise FileNotFoundError(
                f"Ground-truth nodule mask not found for {sample.sample_id} under {args.processed_dir}"
            )
        visible_lung_coverage = gt_inside_visible_lung_coverage(
            gt_overlay.mask,
            output.display_image,
            args.visible_lung_threshold,
        )
        if (
            args.require_gt_inside_visible_lung
            and gt_overlay.available
            and visible_lung_coverage < args.gt_visible_lung_min_coverage
        ):
            continue
        gt_hit = gradcam_hits_ground_truth(
            output.heatmap,
            gt_overlay.mask,
            args.gt_hit_cam_percentile,
            args.gt_hit_min_mask_coverage,
        )
        if args.require_malignant_gt_hit and int(row["label"]) == CLASS_NAME_TO_INDEX["malignant"] and not gt_hit["hit"]:
            continue

        stem = safe_name(
            f"fold{args.fold}_{args.split}_{sample.sample_id}_true-{sample.class_name}"
            f"_pred-{args.classes[output.predicted_class]}_target-{args.classes[target_class]}"
        )
        image_path = args.output_dir / "images" / f"{stem}_image.png"
        gt_path = args.output_dir / "ground_truth" / f"{stem}_gt_nodule.png"
        heatmap_path = args.output_dir / "heatmaps" / f"{stem}_heatmap.png"
        overlay_path = args.output_dir / "overlays" / f"{stem}_overlay.png"
        figure_path = args.output_dir / "figures" / f"{stem}_figure.png"
        for path in (image_path, gt_path, heatmap_path, overlay_path, figure_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        save_image(output.display_image, image_path)
        save_image(gt_overlay.image, gt_path)
        save_heatmap(output.heatmap, heatmap_path, args.cmap)
        save_overlay(output.display_image, output.heatmap, overlay_path, args.alpha, args.cmap)
        panel = make_summary_panel(
            output=output,
            classes=args.classes,
            sample_id=sample.sample_id,
            true_class=int(row["label"]),
            alpha=args.alpha,
            cmap=args.cmap,
            gt_overlay=gt_overlay,
            layout=args.figure_layout,
        )
        panel.save(figure_path)
        panels.append(panel)

        records.append(
            {
                "sample_id": sample.sample_id,
                "dataset_index": int(row["dataset_index"]),
                "source_path": str(sample.path),
                "fold": args.fold,
                "split": args.split,
                "label": int(row["label"]),
                "label_name": sample.class_name,
                "prediction": output.predicted_class,
                "prediction_name": args.classes[output.predicted_class],
                "score": float(row["score"]),
                "target_class": target_class,
                "target_name": args.classes[target_class],
                "score_benign": float(output.probabilities[0].item()),
                "score_malignant": float(output.probabilities[1].item()) if len(args.classes) > 1 else np.nan,
                "correct": bool(output.predicted_class == int(row["label"])),
                "gt_mask_available": gt_overlay.available,
                "gt_mask_description": gt_overlay.description,
                "gt_clipped_to_lung": gt_overlay.clipped_to_lung,
                "gt_removed_outside_lung_voxels": gt_overlay.removed_outside_lung_voxels,
                "gt_visible_lung_coverage": visible_lung_coverage,
                "gt_hit": gt_hit["hit"],
                "gt_hotspot_mask_coverage": gt_hit["mask_coverage"],
                "gt_hotspot_precision": gt_hit["hotspot_precision"],
                "gt_hit_cam_percentile": args.gt_hit_cam_percentile,
                "image_path": str(image_path),
                "gt_overlay_path": str(gt_path),
                "heatmap_path": str(heatmap_path),
                "overlay_path": str(overlay_path),
                "figure_path": str(figure_path),
            }
        )

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise RuntimeError("No Grad-CAM figures remained after applying the requested filters.")
    if args.require_malignant_gt_hit:
        manifest = select_manifest_rows(
            manifest,
            args.selection,
            args.max_samples,
            args.balance_fill_shortfall,
        )
        panels = [Image.open(path).convert("RGB") for path in manifest["figure_path"].tolist()]
    if manifest.empty:
        raise RuntimeError("No Grad-CAM figures remained after applying the requested filters.")

    manifest_path = args.output_dir / "gradcam_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary_path = ""
    if not args.no_summary:
        summary_path = str(args.output_dir / "gradcam_summary.png")
        make_summary_mosaic(panels, args.summary_cols).save(summary_path)
    write_report(args.output_dir / "gradcam_report.md", manifest, Path(summary_path) if summary_path else None)
    print(f"Wrote {len(manifest)} Grad-CAM figures to {args.output_dir}")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_run_config(run_dir: Path | None, checkpoint_path: Path) -> dict[str, object]:
    candidates = []
    if run_dir is not None:
        candidates.append(run_dir / "config.json")
    candidates.extend(parent / "config.json" for parent in checkpoint_path.resolve().parents[:4])
    for candidate in candidates:
        if candidate.exists():
            with candidate.open() as handle:
                return json.load(handle)
    return {}


def normalize_image_size(value: object) -> list[int]:
    if isinstance(value, int):
        return [value, value]
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1:
        return [int(value[0]), int(value[0])]
    if isinstance(value, list) and len(value) >= 2:
        return [int(value[0]), int(value[1])]
    raise ValueError(f"Invalid image size: {value}")


def build_explain_transforms(image_size: list[int]):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((int(image_size[0]), int(image_size[1]))),
            transforms.ToTensor(),
            MinMaxScale(),
        ]
    )


def load_model(
    checkpoint_path: Path,
    backbone: str,
    classes: list[str],
    run_config: dict[str, object],
    device: torch.device,
) -> nn.Module:
    if __package__ is None or __package__ == "":
        from src.luna16_synthetic_2d.models import build_model
    else:
        from .models import build_model

    output_dim = 1 if len(classes) == 2 else len(classes)
    model = build_model(
        backbone=backbone,
        num_classes=output_dim,
        pretrained=False,
        freeze_backbone=bool(run_config.get("freeze_backbone", False)),
        freeze_half_backbone=bool(run_config.get("freeze_half_backbone", False)),
        freeze_first_layers=int(run_config.get("freeze_first_layers", 0)),
        unfreeze_last_layers=int(run_config.get("unfreeze_last_layers", 0)),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state_dict = strip_state_dict_prefix(state_dict, "model.")
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def strip_state_dict_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not all(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix) :]: value for key, value in state_dict.items()}


def score_dataset(model: nn.Module, dataset: SyntheticLuna16Dataset, device: torch.device) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            tensor, label, sample_id = dataset[index]
            logits = model(tensor.unsqueeze(0).to(device))
            probabilities = probabilities_from_logits(logits).squeeze(0).cpu()
            prediction = int(probabilities.argmax().item())
            true_class_score = float(probabilities[int(label)].item())
            records.append(
                {
                    "dataset_index": index,
                    "sample_id": str(sample_id),
                    "label": int(label),
                    "prediction": prediction,
                    "score": float(probabilities[prediction].item()),
                    "true_class_score": true_class_score,
                    "correct": prediction == int(label),
                }
            )
    return pd.DataFrame(records)


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[1] == 1:
        positive = torch.sigmoid(logits[:, 0])
        return torch.stack([1.0 - positive, positive], dim=1)
    return torch.softmax(logits, dim=1)


def select_rows(
    rows: pd.DataFrame,
    sample_ids: list[str] | None,
    selection: str,
    max_samples: int,
    balance_fill_shortfall: bool,
) -> pd.DataFrame:
    selected = rows.copy()
    if sample_ids:
        wanted = {str(sample_id) for sample_id in sample_ids}
        selected = selected[selected["sample_id"].astype(str).isin(wanted)].copy()

    if selection.startswith("balanced_"):
        inner_selection = selection.removeprefix("balanced_")
        return select_balanced_rows(selected, inner_selection, max_samples, balance_fill_shortfall)

    if selection == "correct":
        selected = selected[selected["correct"]].copy()
    elif selection == "incorrect":
        selected = selected[~selected["correct"]].copy()
    elif selection == "highest_score":
        selected = selected.sort_values("score", ascending=False)
    elif selection == "lowest_score":
        selected = selected.sort_values("score", ascending=True)
    else:
        selected = selected.sort_values("dataset_index")
    return selected.head(max(1, int(max_samples))).reset_index(drop=True)


def exclude_sample_ids(rows: pd.DataFrame, exclude_ids: list[str] | None) -> pd.DataFrame:
    if not exclude_ids:
        return rows
    excluded = {normalize_sample_id(sample_id) for sample_id in exclude_ids}
    keep_mask = ~rows["sample_id"].map(lambda value: normalize_sample_id(value) in excluded)
    return rows[keep_mask].copy().reset_index(drop=True)


def normalize_sample_id(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).strip()


def select_balanced_rows(
    rows: pd.DataFrame,
    selection: str,
    max_samples: int,
    fill_shortfall: bool,
) -> pd.DataFrame:
    max_samples = max(1, int(max_samples))
    labels = sorted(rows["label"].dropna().astype(int).unique().tolist())
    if not labels:
        return rows.head(max_samples).reset_index(drop=True)

    requested_quota = max(1, max_samples // len(labels))
    available_per_label = {
        label: int((rows["label"].astype(int) == label).sum())
        for label in labels
    }
    strict_quota = min(requested_quota, min(available_per_label.values()))
    quota_by_label = {label: strict_quota for label in labels}

    if fill_shortfall:
        base_quota = max_samples // len(labels)
        remainder = max_samples % len(labels)
        quota_by_label = {
            label: base_quota + (1 if label_index < remainder else 0)
            for label_index, label in enumerate(labels)
        }

    selected_frames: list[pd.DataFrame] = []
    for label in labels:
        label_rows = rows[rows["label"].astype(int) == label].copy()
        label_rows = sort_for_selection(label_rows, selection)
        selected_frames.append(label_rows.head(quota_by_label[label]))

    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    if fill_shortfall and len(selected) < max_samples:
        selected_indices = set(selected["dataset_index"].astype(int).tolist()) if not selected.empty else set()
        leftovers = rows[~rows["dataset_index"].astype(int).isin(selected_indices)].copy()
        leftovers = sort_for_selection(leftovers, selection)
        selected = pd.concat([selected, leftovers.head(max_samples - len(selected))], ignore_index=True)

    return selected.sort_values(["label", "dataset_index"]).head(max_samples).reset_index(drop=True)


def sort_for_selection(rows: pd.DataFrame, selection: str) -> pd.DataFrame:
    if selection == "highest_score":
        return rows.sort_values("score", ascending=False)
    if selection == "lowest_score":
        return rows.sort_values("score", ascending=True)
    if selection == "correct":
        return rows[rows["correct"]].sort_values("dataset_index")
    if selection == "incorrect":
        return rows[~rows["correct"]].sort_values("dataset_index")
    return rows.sort_values("dataset_index")


def resolve_target_class(target_class: str, true_class: int, predicted_class: int) -> int:
    if target_class == "true":
        return true_class
    if target_class == "predicted":
        return predicted_class
    return CLASS_NAME_TO_INDEX[target_class]


def compute_gradcam(
    model: nn.Module,
    tensor: torch.Tensor,
    target_class: int,
    target_layer_name: str | None,
) -> GradCamOutput:
    tensor = tensor.detach().clone().requires_grad_(True)
    target_layer = get_target_layer(model, target_layer_name)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def save_activation(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output)

    def save_gradient(_module: nn.Module, _grad_inputs: tuple[torch.Tensor, ...], grad_outputs: tuple[torch.Tensor, ...]) -> None:
        gradients.append(grad_outputs[0])

    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        probabilities = probabilities_from_logits(logits).squeeze(0).detach().cpu()
        predicted_class = int(probabilities.argmax().item())
        objective = class_objective(logits, target_class)
        objective.backward()
        heatmap = gradcam_from_tensors(activations[-1], gradients[-1], tensor.shape[-2:])
    finally:
        forward_handle.remove()
        backward_handle.remove()

    display_image = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    display_image = np.clip(display_image, 0.0, 1.0)
    return GradCamOutput(
        display_image=display_image,
        heatmap=heatmap,
        logits=logits.detach().cpu(),
        probabilities=probabilities,
        predicted_class=predicted_class,
        target_class=target_class,
    )


def get_target_layer(module: nn.Module, target_layer_name: str | None) -> nn.Conv2d:
    if target_layer_name:
        named_modules = dict(module.named_modules())
        if target_layer_name not in named_modules:
            raise ValueError(f"Target layer '{target_layer_name}' not found in model.")
        target = named_modules[target_layer_name]
        if not isinstance(target, nn.Conv2d):
            raise TypeError(f"Target layer '{target_layer_name}' is {type(target).__name__}, not Conv2d.")
        return target

    last_conv: nn.Conv2d | None = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            last_conv = child
    if last_conv is None:
        raise ValueError("Could not find a Conv2d layer for Grad-CAM.")
    return last_conv


def class_objective(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[1] == 1:
        positive_logit = logits[:, 0]
        objective = positive_logit if target_class == 1 else -positive_logit
        return objective.sum()
    return logits[:, target_class].sum()


def gradcam_from_tensors(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    output_size: tuple[int, int],
) -> np.ndarray:
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
    cam_array = cam.squeeze().detach().cpu().numpy()
    return normalize(cam_array)


def normalize(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array).astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def save_image(image: np.ndarray, output_path: Path) -> None:
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).save(output_path)


def save_heatmap(heatmap: np.ndarray, output_path: Path, cmap: str) -> None:
    colored = plt.get_cmap(cmap)(heatmap)[..., :3]
    save_image(colored, output_path)


def save_overlay(image: np.ndarray, heatmap: np.ndarray, output_path: Path, alpha: float, cmap: str) -> None:
    overlay = overlay_array(image, heatmap, alpha, cmap)
    save_image(overlay, output_path)


def overlay_array(image: np.ndarray, heatmap: np.ndarray, alpha: float, cmap: str) -> np.ndarray:
    colored = plt.get_cmap(cmap)(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * colored, 0.0, 1.0)


def build_ground_truth_overlay(
    sample_id: str,
    processed_dir: Path,
    display_image: np.ndarray,
    enabled: bool,
    strict: bool,
    clip_to_lung: bool,
) -> GroundTruthOverlay:
    if not enabled:
        return GroundTruthOverlay(display_image, False, "disabled")

    mask_path = find_processed_file(processed_dir, sample_id, "nodule_mask")
    if mask_path is None:
        return GroundTruthOverlay(display_image, False, "missing nodule mask")

    try:
        mask = load_mask_zyx(mask_path) > 0
    except ImportError:
        if strict:
            raise
        return GroundTruthOverlay(display_image, False, "nibabel missing")
    if not mask.any():
        return GroundTruthOverlay(display_image, False, "empty nodule mask")

    clipped_to_lung = False
    removed_outside_lung_voxels = 0
    lung_path = find_processed_file(processed_dir, sample_id, "lung_mask")
    if clip_to_lung and lung_path is not None:
        try:
            lung_mask = load_mask_zyx(lung_path) > 0
        except ImportError:
            if strict:
                raise
            lung_mask = None
        if lung_mask is not None and lung_mask.shape == mask.shape and lung_mask.any():
            before = int(mask.sum())
            mask = mask & lung_mask
            after = int(mask.sum())
            clipped_to_lung = True
            removed_outside_lung_voxels = before - after
            if not mask.any():
                return GroundTruthOverlay(
                    display_image,
                    False,
                    f"nodule mask outside lung after clipping ({mask_path})",
                    None,
                    True,
                    removed_outside_lung_voxels,
                )

    mask_projection = mask.max(axis=0)
    mask_resized = resize_mask(mask_projection, display_image.shape[:2])
    gt_image = draw_mask_on_image(display_image, mask_resized)
    description = str(mask_path)
    if clipped_to_lung:
        description += f" | clipped to lung, removed {removed_outside_lung_voxels} voxels"
    return GroundTruthOverlay(
        gt_image,
        True,
        description,
        mask_resized,
        clipped_to_lung,
        removed_outside_lung_voxels,
    )


def find_processed_file(processed_dir: Path, sample_id: str, kind: str) -> Path | None:
    candidates = [
        processed_dir / sample_id / f"{sample_id}_{kind}.nii.gz",
        processed_dir / sample_id / f"{sample_id}_{kind}.npy",
        processed_dir / sample_id / f"{kind}.nii.gz",
        processed_dir / sample_id / f"{kind}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    patterns = [
        f"**/{sample_id}/{sample_id}_{kind}.nii.gz",
        f"**/{sample_id}/{sample_id}_{kind}.npy",
        f"**/{sample_id}/{kind}.nii.gz",
        f"**/{sample_id}/{kind}.npy",
    ]
    for pattern in patterns:
        matches = sorted(processed_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_mask_zyx(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        array = np.load(path).astype(np.float32)
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D NumPy mask at {path}, got shape {array.shape}.")
        return array
    return load_nifti_zyx(path)


def load_nifti_zyx(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Ground-truth overlays from NIfTI masks require nibabel.") from exc

    array = nib.load(str(path)).get_fdata().astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI mask at {path}, got shape {array.shape}.")
    return array.transpose(2, 1, 0)


def resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    height, width = int(target_hw[0]), int(target_hw[1])
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def draw_mask_on_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB")
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    red = Image.new("RGB", base.size, (255, 20, 20))
    base = Image.composite(Image.blend(base, red, 0.30), base, mask_img)
    draw = ImageDraw.Draw(base)
    edge = mask ^ binary_erode(mask)
    ys, xs = np.where(edge)
    for x, y in zip(xs.tolist(), ys.tolist()):
        draw.point((x, y), fill=(255, 235, 0))
    return np.asarray(base).astype(np.float32) / 255.0


def gradcam_hits_ground_truth(
    heatmap: np.ndarray,
    mask: np.ndarray | None,
    cam_percentile: float,
    min_mask_coverage: float,
) -> dict[str, object]:
    if mask is None or not mask.any():
        return {"hit": False, "mask_coverage": 0.0, "hotspot_precision": 0.0}

    threshold = float(np.percentile(heatmap, cam_percentile))
    hotspot = heatmap >= threshold
    if not hotspot.any():
        return {"hit": False, "mask_coverage": 0.0, "hotspot_precision": 0.0}

    overlap = hotspot & mask
    mask_coverage = float(overlap.sum() / max(1, mask.sum()))
    hotspot_precision = float(overlap.sum() / max(1, hotspot.sum()))
    return {
        "hit": mask_coverage >= float(min_mask_coverage),
        "mask_coverage": mask_coverage,
        "hotspot_precision": hotspot_precision,
    }


def gt_inside_visible_lung_coverage(
    mask: np.ndarray | None,
    display_image: np.ndarray,
    visible_threshold: float,
) -> float:
    if mask is None or not mask.any():
        return 1.0
    grayscale = display_image.mean(axis=2) if display_image.ndim == 3 else display_image
    visible = grayscale > float(visible_threshold)
    visible = binary_dilate(visible, iterations=3)
    return float((mask & visible).sum() / max(1, mask.sum()))


def select_manifest_rows(
    manifest: pd.DataFrame,
    selection: str,
    max_samples: int,
    balance_fill_shortfall: bool,
) -> pd.DataFrame:
    selected = select_rows(
        manifest.copy(),
        sample_ids=None,
        selection=selection,
        max_samples=max_samples,
        balance_fill_shortfall=balance_fill_shortfall,
    )
    return selected.reset_index(drop=True)


def binary_erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbors.append(padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]])
    return np.logical_and.reduce(neighbors)


def binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(max(1, int(iterations))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbors.append(padded[1 + dy : 1 + dy + result.shape[0], 1 + dx : 1 + dx + result.shape[1]])
        result = np.logical_or.reduce(neighbors)
    return result


def make_summary_panel(
    output: GradCamOutput,
    classes: list[str],
    sample_id: str,
    true_class: int,
    alpha: float,
    cmap: str,
    gt_overlay: GroundTruthOverlay,
    layout: str,
) -> Image.Image:
    gt_title = "Ground truth nodule" if gt_overlay.available else f"Ground truth ({gt_overlay.description})"
    gt = image_with_title(gt_overlay.image, gt_title[:80])
    overlay = image_with_title(
        overlay_array(output.display_image, output.heatmap, alpha, cmap),
        f"Model attention overlay: {classes[output.target_class]}",
    )
    if layout == "full":
        image = image_with_title(output.display_image, "Input")
        heatmap = image_with_title(plt.get_cmap(cmap)(output.heatmap)[..., :3], f"Grad-CAM: {classes[output.target_class]}")
        panels = [image, gt, heatmap, overlay]
    else:
        panels = [gt, overlay]
    footer = (
        f"{sample_id} | true {classes[true_class]} | pred {classes[output.predicted_class]} "
        f"| P(mal) {float(output.probabilities[1].item()):.3f}"
    )
    width = sum(panel.width for panel in panels)
    panel_height = max(panel.height for panel in panels)
    height = panel_height + 28
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, panel_height, width, height), fill=(20, 20, 20))
    draw.text((8, panel_height + 7), footer[:180], fill=(255, 255, 255), font=ImageFont.load_default())
    return canvas


def image_with_title(image: np.ndarray, title: str) -> Image.Image:
    base = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB")
    canvas = Image.new("RGB", (base.width, base.height + 28), (255, 255, 255))
    canvas.paste(base, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 28), fill=(255, 255, 255))
    draw.line((0, 27, canvas.width, 27), fill=(210, 210, 210), width=1)
    draw.text((8, 7), title, fill=(20, 20, 20), font=ImageFont.load_default())
    return canvas


def make_summary_mosaic(panels: list[Image.Image], columns: int) -> Image.Image:
    columns = max(1, int(columns))
    rows = math.ceil(len(panels) / columns)
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (columns * width, rows * height), (245, 245, 245))
    for index, panel in enumerate(panels):
        x = (index % columns) * width
        y = (index // columns) * height
        canvas.paste(panel, (x, y))
    return canvas


def write_report(report_path: Path, manifest: pd.DataFrame, summary_path: Path | None) -> None:
    class_counts = manifest["label_name"].value_counts().sort_index()
    lines = [
        "# LUNA16 Synthetic 2D Grad-CAM",
        "",
        f"- Manifest: `{manifest_path_relative(report_path, 'gradcam_manifest.csv')}`",
        f"- Figures: {len(manifest)}",
        "- Class counts: " + ", ".join(f"{label}={count}" for label, count in class_counts.items()),
    ]
    if summary_path is not None:
        lines.extend(["", f'<img src="{summary_path.name}" width="1100">', ""])
    for _, row in manifest.iterrows():
        figure_rel = Path(str(row["figure_path"])).relative_to(report_path.parent).as_posix()
        lines.extend(
            [
                "",
                f"## {row['sample_id']}",
                "",
                (
                    f"True `{row['label_name']}`, predicted `{row['prediction_name']}`, "
                    f"Grad-CAM target `{row['target_name']}`."
                ),
                "",
                f'<img src="{figure_rel}" width="900">',
            ]
        )
    report_path.write_text("\n".join(lines).rstrip() + "\n")


def manifest_path_relative(report_path: Path, filename: str) -> str:
    return (report_path.parent / filename).relative_to(report_path.parent).as_posix()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    main()
