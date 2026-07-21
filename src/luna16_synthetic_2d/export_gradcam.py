from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .datamodule import build_transforms
from .dataset import load_image
from .lightning_model import SyntheticLuna16Classifier


DEFAULT_SELECTION_CSV = Path("docs/luna16_synthetic_2d_class_rankings/luna16_synthetic_2d_class_top_bottom.csv")
DEFAULT_OUTPUT_DIR = Path("docs/luna16_synthetic_2d_gradcam")
DEFAULT_EXPERIMENTS = (
    "luna16_synthetic_2d_top5_minprob0.5",
    "luna16_synthetic_2d_top7_minprob0.3_rbf",
    "luna16_synthetic_2d_top7_minprob0.3_shepard",
    "luna16_synthetic_2d_top4_minprob0.5_rbf",
    "luna16_synthetic_2d_top4_minprob0.5_shepard",
    "luna16_synthetic_2d_top3_minprob0.5_rbf",
    "luna16_synthetic_2d_top3_minprob0.5_shepard",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GradCAM overlays for LUNA16 synthetic 2D classifiers.")
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--backbones", nargs="+", default=None)
    parser.add_argument("--rank-kinds", nargs="+", default=["top", "bottom"], choices=["top", "bottom", "all"])
    parser.add_argument("--labels", nargs="+", default=["benign", "malignant"])
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--target-class", choices=["predicted", "true", "malignant", "benign"], default="predicted")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--report-max-rows",
        type=int,
        default=1000,
        help="Maximum GradCAM rows to embed in the markdown report. Use -1 to include all rows.",
    )
    parser.add_argument(
        "--synthetic-root-base",
        type=Path,
        default=None,
        help=(
            "Optional server root containing synthetic image folders. If an image_path from the "
            "selection CSV starts with data/ and does not exist, it is remapped under this root."
        ),
    )
    parser.add_argument("--checkpoint-glob", default="*.ckpt")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-missing-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    rows = load_selection(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_cache: dict[tuple[str, int, str], SyntheticLuna16Classifier] = {}
    manifest_rows: list[dict[str, object]] = []
    missing_checkpoints: list[str] = []

    for row_index, row in rows.iterrows():
        experiment = str(row["experiment"])
        fold = int(row["fold"])
        backbone = str(row["backbone"])
        sample_id = str(row["sample_id"])
        target_label = target_label_from_row(row, args.target_class)
        target_name = "malignant" if target_label == 1 else "benign"
        out_stem = safe_name(
            f"{experiment}_{backbone}_fold{fold}_{row['label_name']}_{row['rank_kind']}"
            f"{int(row['rank_order']):02d}_{sample_id}_{args.target_class}-{target_name}"
        )
        overlay_path = args.output_dir / "overlays" / f"{out_stem}.png"
        heatmap_path = args.output_dir / "heatmaps" / f"{out_stem}.png"

        if args.skip_existing and overlay_path.exists() and heatmap_path.exists():
            manifest_rows.append(manifest_row(row, target_name, overlay_path, heatmap_path, "skipped_existing"))
            continue

        cache_key = (experiment, fold, backbone)
        if cache_key not in model_cache:
            checkpoint_path = find_checkpoint(Path("outputs") / experiment / f"fold_{fold}" / backbone, args.checkpoint_glob)
            if checkpoint_path is None:
                message = f"{experiment}/fold_{fold}/{backbone}"
                missing_checkpoints.append(message)
                if args.fail_missing_checkpoint:
                    raise FileNotFoundError(f"No checkpoint found for {message}")
                manifest_rows.append(manifest_row(row, target_name, overlay_path, heatmap_path, "missing_checkpoint"))
                continue
            model_cache[cache_key] = load_model_from_run(
                output_dir=Path("outputs") / experiment / f"fold_{fold}" / backbone,
                checkpoint_path=checkpoint_path,
                device=device,
            )

        model = model_cache[cache_key]
        image_path = resolve_synthetic_image_path(Path(str(row["image_path"])), args.synthetic_root_base)
        image_size = image_size_from_run(Path("outputs") / experiment / f"fold_{fold}" / backbone)
        result = compute_gradcam(model, image_path=image_path, image_size=image_size, target_label=target_label, device=device)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        save_overlay(result.image, result.cam, overlay_path, title=overlay_title(row, target_name))
        save_heatmap(result.cam, heatmap_path)
        manifest_rows.append(
            manifest_row(row, target_name, overlay_path, heatmap_path, "ok", result.predicted_score, image_path)
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "gradcam_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    write_report(args.output_dir / "gradcam_report.md", manifest, missing_checkpoints, args.report_max_rows)


def load_selection(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(args.selection_csv)
    if "rank_kind" not in frame.columns:
        frame["rank_kind"] = "all"
    if "rank_order" not in frame.columns:
        if "rank_within_true_class" not in frame.columns:
            raise RuntimeError(
                f"{args.selection_csv} must contain rank_order or rank_within_true_class"
            )
        frame["rank_order"] = frame["rank_within_true_class"]
    frame = frame[frame["experiment"].isin(args.experiments)].copy()
    frame = frame[frame["rank_kind"].isin(args.rank_kinds)].copy()
    frame = frame[frame["label_name"].isin(args.labels)].copy()
    frame = frame[pd.to_numeric(frame["rank_order"], errors="coerce") <= args.top_n].copy()
    if args.backbones is not None:
        frame = frame[frame["backbone"].isin(args.backbones)].copy()
    if frame.empty:
        raise RuntimeError(f"No rows selected from {args.selection_csv}")
    return frame.sort_values(["experiment", "backbone", "fold", "label_name", "rank_kind", "rank_order"])


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_synthetic_image_path(image_path: Path, synthetic_root_base: Path | None) -> Path:
    if image_path.exists() or synthetic_root_base is None:
        return image_path
    parts = image_path.parts
    if not parts or parts[0] != "data" or len(parts) < 2:
        return image_path
    remapped = synthetic_root_base.joinpath(*parts[1:])
    if remapped.exists():
        return remapped
    return image_path


def find_checkpoint(run_dir: Path, checkpoint_glob: str) -> Path | None:
    checkpoint_dir = run_dir / "checkpoints"
    candidates = sorted(checkpoint_dir.glob(checkpoint_glob))
    candidates = [path for path in candidates if path.is_file() and not path.name.startswith("last")]
    if not candidates:
        candidates = sorted(path for path in checkpoint_dir.glob("*.ckpt") if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_model_from_run(output_dir: Path, checkpoint_path: Path, device: torch.device) -> SyntheticLuna16Classifier:
    config = load_config(output_dir)
    model = SyntheticLuna16Classifier(
        backbone=str(config.get("backbone", output_dir.name)),
        num_classes=len(config.get("classes", ["benign", "malignant"])),
        class_names=list(config.get("classes", ["benign", "malignant"])),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        pretrained=False,
        freeze_backbone=bool(config.get("freeze_backbone", False)),
        freeze_half_backbone=bool(config.get("freeze_half_backbone", False)),
        freeze_first_layers=int(config.get("freeze_first_layers", 0)),
        unfreeze_last_layers=int(config.get("unfreeze_last_layers", 0)),
        max_epochs=int(config.get("epochs", 100)),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


def load_config(output_dir: Path) -> dict[str, object]:
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open() as handle:
        return json.load(handle)


def image_size_from_run(output_dir: Path) -> list[int]:
    config = load_config(output_dir)
    image_size = config.get("image_size", [256, 384])
    if isinstance(image_size, int):
        return [image_size, image_size]
    return list(image_size)


def target_label_from_row(row: pd.Series, target_class: str) -> int:
    if target_class == "malignant":
        return 1
    if target_class == "benign":
        return 0
    column = "prediction" if target_class == "predicted" else "label"
    return int(row[column])


class GradcamResult:
    def __init__(self, image: np.ndarray, cam: np.ndarray, predicted_score: float) -> None:
        self.image = image
        self.cam = cam
        self.predicted_score = predicted_score


def compute_gradcam(
    model: SyntheticLuna16Classifier,
    image_path: Path,
    image_size: list[int],
    target_label: int,
    device: torch.device,
) -> GradcamResult:
    transform = build_transforms(image_size, train=False)
    image = load_image(image_path)
    tensor = transform(image).unsqueeze(0).to(device)
    display_image = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    display_image = np.clip(display_image, 0.0, 1.0)

    target_layer = find_last_conv2d(model.model)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_inputs, grad_outputs):
        gradients.append(grad_outputs[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        if logits.ndim == 2 and logits.shape[1] == 1:
            positive_logit = logits[:, 0]
            predicted_score = float(torch.sigmoid(positive_logit).item())
            target = positive_logit if target_label == 1 else -positive_logit
            target.sum().backward()
        else:
            probs = torch.softmax(logits, dim=1)
            predicted_score = float(probs[:, target_label].item())
            logits[:, target_label].sum().backward()

        activation = activations[-1].detach()
        gradient = gradients[-1].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = normalize_cam(cam)
    finally:
        forward_handle.remove()
        backward_handle.remove()

    return GradcamResult(image=display_image, cam=cam, predicted_score=predicted_score)


def find_last_conv2d(module: nn.Module) -> nn.Conv2d:
    last_conv: nn.Conv2d | None = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            last_conv = child
    if last_conv is None:
        raise ValueError("Could not find a Conv2d layer for GradCAM")
    return last_conv


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.nan_to_num(cam)
    min_value = float(cam.min())
    max_value = float(cam.max())
    if max_value <= min_value:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - min_value) / (max_value - min_value)).astype(np.float32)


def save_overlay(image: np.ndarray, cam: np.ndarray, output_path: Path, title: str) -> None:
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    overlay = np.clip(0.58 * image + 0.42 * heatmap, 0.0, 1.0)
    canvas = Image.fromarray((overlay * 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 28), fill=(0, 0, 0))
    draw.text((6, 7), title[:120], fill=(255, 255, 255))
    canvas.save(output_path)


def save_heatmap(cam: np.ndarray, output_path: Path) -> None:
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    Image.fromarray((heatmap * 255).astype(np.uint8)).save(output_path)


def overlay_title(row: pd.Series, target_name: str) -> str:
    return (
        f"{row['experiment']} | {row['backbone']} | fold {int(row['fold'])} | "
        f"{row['label_name']} {row['rank_kind']} {int(row['rank_order'])} | target {target_name}"
    )


def manifest_row(
    row: pd.Series,
    target_name: str,
    overlay_path: Path,
    heatmap_path: Path,
    status: str,
    gradcam_score: float | None = None,
    resolved_image_path: Path | None = None,
) -> dict[str, object]:
    result = row.to_dict()
    result["gradcam_target_name"] = target_name
    result["gradcam_status"] = status
    result["gradcam_score"] = gradcam_score
    result["resolved_image_path"] = str(resolved_image_path) if resolved_image_path is not None else ""
    result["gradcam_overlay_path"] = str(overlay_path)
    result["gradcam_heatmap_path"] = str(heatmap_path)
    return result


def write_report(
    report_path: Path,
    manifest: pd.DataFrame,
    missing_checkpoints: list[str],
    report_max_rows: int,
) -> None:
    ok = manifest[manifest["gradcam_status"] == "ok"].copy()
    report_rows = ok if report_max_rows < 0 else ok.head(report_max_rows)
    lines = [
        "# LUNA16 Synthetic 2D GradCAM",
        "",
        f"- Manifest: `{report_path.parent / 'gradcam_manifest.csv'}`",
        f"- Overlay generated: {len(ok)} / {len(manifest)}",
    ]
    if report_max_rows >= 0 and len(ok) > len(report_rows):
        lines.append(f"- Report preview rows: {len(report_rows)} / {len(ok)}")
        lines.append("- Full image list is available in the manifest CSV.")
    lines.append("")
    if missing_checkpoints:
        lines.extend(["## Missing Checkpoints", ""])
        for item in sorted(set(missing_checkpoints)):
            lines.append(f"- `{item}`")
        lines.append("")

    for experiment in sorted(report_rows["experiment"].unique()):
        lines.extend(["", f"## {experiment}", ""])
        experiment_rows = report_rows[report_rows["experiment"] == experiment]
        for label_name in sorted(experiment_rows["label_name"].unique()):
            label_rows = experiment_rows[experiment_rows["label_name"] == label_name]
            lines.extend(["", f"### {label_name}", ""])
            table = label_rows[
                [
                    "rank_kind",
                    "rank_order",
                    "backbone",
                    "fold",
                    "sample_id",
                    "prediction_name",
                    "score",
                    "true_class_score",
                    "gradcam_target_name",
                    "gradcam_overlay_path",
                ]
            ].copy()
            table["gradcam_overlay_path"] = table["gradcam_overlay_path"].map(
                lambda value: f'<img src="{Path(value).relative_to(report_path.parent).as_posix()}" width="180">'
            )
            lines.extend(dataframe_to_markdown(table))
            lines.append("")

    report_path.write_text("\n".join(lines).rstrip() + "\n")


def dataframe_to_markdown(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]
    for _, row in frame.iterrows():
        values = [markdown_cell(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return rows


def markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        value = f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    main()
