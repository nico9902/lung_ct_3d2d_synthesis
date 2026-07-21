from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi


DEFAULT_MANIFEST = Path("docs/luna16_synthetic_2d_gradcam_all_predicted/gradcam_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("docs/luna16_synthetic_2d_gradcam_comparison_mosaics_predicted")
DEFAULT_COLUMNS = (
    "luna16_synthetic_2d_gt",
    "luna16_synthetic_2d_top3_minprob0.5_rbf",
    "luna16_synthetic_2d_top4_minprob0.5_rbf",
    "luna16_synthetic_2d_top5_minprob0.5",
    "luna16_synthetic_2d_top7_minprob0.3_rbf",
)
EXPERIMENT_LABELS = {
    "luna16_synthetic_2d_gt": "Synth GT",
    "luna16_synthetic_2d_top3_minprob0.5_rbf": "Top3 RBF",
    "luna16_synthetic_2d_top3_minprob0.5_shepard": "Top3 Shepard",
    "luna16_synthetic_2d_top4_minprob0.5_rbf": "Top4 RBF",
    "luna16_synthetic_2d_top4_minprob0.5_shepard": "Top4 Shepard",
    "luna16_synthetic_2d_top5_minprob0.5": "Top5",
    "luna16_synthetic_2d_top7_minprob0.3_rbf": "Top7 RBF",
    "luna16_synthetic_2d_top7_minprob0.3_shepard": "Top7 Shepard",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create multi-column mosaics with CT GT, synthetic GradCAM overlays, and detector nodule-hit labels."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/LUNA16_preprocessed"))
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_COLUMNS))
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "gt_correct_top5_wrong",
            "gt_wrong_top5_correct",
            "both_correct",
            "both_wrong",
            "top5_false_positive",
            "top5_false_negative",
        ],
    )
    parser.add_argument("--max-samples-per-category", type=int, default=60)
    parser.add_argument("--rows-per-page", type=int, default=4)
    parser.add_argument("--panel-width", type=int, default=240)
    parser.add_argument("--panel-height", type=int, default=260)
    parser.add_argument("--ct-crop-size", type=int, default=128)
    parser.add_argument("--ct-margin", type=int, default=32)
    parser.add_argument("--hit-margin-voxels", type=float, default=8.0)
    parser.add_argument("--window-center", type=float, default=-600.0)
    parser.add_argument("--window-width", type=float, default=1500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[(manifest["gradcam_status"] == "ok") & (manifest["backbone"] == args.backbone)].copy()
    if manifest.empty:
        raise RuntimeError(f"No usable GradCAM rows found in {args.manifest}")

    sample_table = build_sample_table(manifest, args.experiments)
    if sample_table.empty:
        raise RuntimeError("No samples contain all requested experiment overlays.")

    volume_cache: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, object]] = []
    row_records: list[dict[str, object]] = []

    for category in args.categories:
        rows = select_category(sample_table, category)
        if rows.empty:
            continue
        rows = rows.head(args.max_samples_per_category)
        page_info, records = save_category_pages(rows, category, args, volume_cache)
        pages.extend(page_info)
        row_records.extend(records)

    index = pd.DataFrame(pages)
    index.to_csv(args.output_dir / "comparison_mosaic_index.csv", index=False)
    pd.DataFrame(row_records).to_csv(args.output_dir / "comparison_mosaic_rows.csv", index=False)
    write_report(args.output_dir / "comparison_mosaic_report.md", index, args.output_dir)
    print(f"Wrote {len(index)} comparison mosaic pages to {args.output_dir}")


def build_sample_table(manifest: pd.DataFrame, experiments: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = manifest.groupby("sample_id")
    for sample_id, group in grouped:
        by_experiment = {str(row["experiment"]): row for _, row in group.iterrows()}
        if any(experiment not in by_experiment for experiment in experiments):
            continue
        gt_row = by_experiment["luna16_synthetic_2d_gt"]
        row = {
            "sample_id": sample_id,
            "label": int(gt_row["label"]),
            "label_name": str(gt_row["label_name"]),
            "fold": int(gt_row["fold"]),
        }
        for experiment in experiments:
            exp_row = by_experiment[experiment]
            prefix = safe_column(experiment)
            row[f"{prefix}_prediction_name"] = str(exp_row["prediction_name"])
            row[f"{prefix}_correct"] = bool(exp_row["correct"])
            row[f"{prefix}_true_class_score"] = float(exp_row["true_class_score"])
            row[f"{prefix}_score"] = float(exp_row["score"])
            row[f"{prefix}_overlay_path"] = str(exp_row["gradcam_overlay_path"])
            row[f"{prefix}_image_path"] = str(prefer_existing_path(exp_row.get("resolved_image_path", ""), exp_row["image_path"]))
        row["sort_gap_top5"] = abs(
            row[f"{safe_column('luna16_synthetic_2d_gt')}_true_class_score"]
            - row[f"{safe_column('luna16_synthetic_2d_top5_minprob0.5')}_true_class_score"]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sort_gap_top5", ascending=False)


def select_category(table: pd.DataFrame, category: str) -> pd.DataFrame:
    gt = safe_column("luna16_synthetic_2d_gt")
    top5 = safe_column("luna16_synthetic_2d_top5_minprob0.5")
    gt_ok = table[f"{gt}_correct"].astype(bool)
    top5_ok = table[f"{top5}_correct"].astype(bool)
    benign = table["label_name"] == "benign"
    malignant = table["label_name"] == "malignant"
    top5_pred = table[f"{top5}_prediction_name"]
    categories = {
        "gt_correct_top5_wrong": gt_ok & ~top5_ok,
        "gt_wrong_top5_correct": ~gt_ok & top5_ok,
        "both_correct": gt_ok & top5_ok,
        "both_wrong": ~gt_ok & ~top5_ok,
        "top5_false_positive": benign & (top5_pred == "malignant"),
        "top5_false_negative": malignant & (top5_pred == "benign"),
    }
    if category not in categories:
        raise ValueError(f"Unknown category {category}. Available: {sorted(categories)}")
    return table[categories[category]].copy()


def save_category_pages(
    rows: pd.DataFrame,
    category: str,
    args: argparse.Namespace,
    volume_cache: dict[str, tuple[np.ndarray | None, np.ndarray | None]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output_dir = args.output_dir / safe_name(category)
    sample_dir = args.output_dir / "sample_jpegs" / safe_name(category)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    columns = ["ct_full", "ct_nodule_overlay", *experiment_columns(args.experiments)]
    page_width = len(columns) * args.panel_width
    row_height = args.panel_height
    pages: list[dict[str, object]] = []
    records: list[dict[str, object]] = []

    for page_idx in range(math.ceil(len(rows) / args.rows_per_page)):
        page_rows = rows.iloc[page_idx * args.rows_per_page : (page_idx + 1) * args.rows_per_page]
        canvas = Image.new("RGB", (page_width, row_height * len(page_rows)), (18, 18, 18))
        for row_idx, (_, row) in enumerate(page_rows.iterrows()):
            y = row_idx * row_height
            sample_id = str(row["sample_id"])
            volume, mask = load_volume_mask(sample_id, args.processed_dir, volume_cache)
            hit_info = mask_hit_info(mask)
            sample_canvas = Image.new("RGB", (page_width, row_height), (18, 18, 18))
            sample_canvas.paste(make_ct_panel(sample_id, row, volume, mask, args, overlay=False), (0, 0))
            sample_canvas.paste(make_ct_panel(sample_id, row, volume, mask, args, overlay=True), (args.panel_width, 0))
            record = {
                "category": category,
                "sample_id": sample_id,
                "label_name": row["label_name"],
                "fold": row["fold"],
            }
            col_idx = 2
            for experiment in args.experiments:
                detection_label = "GT mask"
                if experiment != "luna16_synthetic_2d_gt":
                    detection_panel, detection_label = make_detection_panel(
                        row=row,
                        experiment=experiment,
                        volume=volume,
                        mask=mask,
                        args=args,
                    )
                    sample_canvas.paste(detection_panel, (col_idx * args.panel_width, 0))
                    col_idx += 1
                panel, hit_label = make_experiment_panel(row, experiment, hit_info, args)
                sample_canvas.paste(panel, (col_idx * args.panel_width, 0))
                col_idx += 1
                record[f"{safe_column(experiment)}_detections"] = detection_label
                record[f"{safe_column(experiment)}_hit"] = hit_label
            sample_path = sample_dir / f"{safe_name(category)}_{row_idx + 1 + page_idx * args.rows_per_page:03d}_{safe_name(sample_id)}.jpg"
            sample_canvas.save(sample_path, quality=92)
            canvas.paste(sample_canvas, (0, y))
            record["sample_jpeg_path"] = str(sample_path)
            records.append(record)
        path = output_dir / f"{safe_name(category)}_page_{page_idx + 1:03d}.jpg"
        canvas.save(path, quality=92)
        pages.append({"category": category, "page": page_idx + 1, "samples": len(page_rows), "path": str(path)})
    return pages, records


def experiment_columns(experiments: list[str]) -> list[str]:
    columns: list[str] = []
    for experiment in experiments:
        if experiment == "luna16_synthetic_2d_gt":
            columns.append(experiment)
        else:
            columns.extend([f"{experiment}_detections", experiment])
    return columns


def load_volume_mask(
    sample_id: str,
    processed_dir: Path,
    cache: dict[str, tuple[np.ndarray | None, np.ndarray | None]],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if sample_id in cache:
        return cache[sample_id]
    volume_path = processed_dir / sample_id / f"{sample_id}_volume.nii.gz"
    mask_path = processed_dir / sample_id / f"{sample_id}_nodule_mask.nii.gz"
    if not volume_path.exists() or not mask_path.exists():
        cache[sample_id] = (None, None)
        return cache[sample_id]
    volume = nib.load(str(volume_path)).get_fdata().astype(np.float32).transpose(2, 1, 0)
    mask = nib.load(str(mask_path)).get_fdata().astype(np.float32).transpose(2, 1, 0) > 0
    cache[sample_id] = (volume, mask)
    return cache[sample_id]


def make_ct_panel(
    sample_id: str,
    row: pd.Series,
    volume: np.ndarray | None,
    mask: np.ndarray | None,
    args: argparse.Namespace,
    overlay: bool,
) -> Image.Image:
    title = f"CT {'+ nodule' if overlay else 'slice'} | true {row['label_name']}"
    subtitle = f"sample {short_id(sample_id)}"
    if volume is None or mask is None:
        return missing_panel(title, subtitle, args.panel_width, args.panel_height)

    _, slice_2d, mask_slice = full_ct_slice(volume, mask)
    base = Image.fromarray(window_to_uint8(slice_2d, args.window_center, args.window_width), mode="L").convert("RGB")
    if overlay and mask_slice.any():
        mask_img = Image.fromarray((mask_slice.astype(np.uint8) * 255), mode="L")
        red = Image.new("RGB", base.size, (255, 20, 20))
        base = Image.composite(Image.blend(base, red, 0.30), base, mask_img)
        edge = np.logical_xor(mask_slice, ndi.binary_erosion(mask_slice))
        draw = ImageDraw.Draw(base)
        edge_y, edge_x = np.where(edge)
        for x, y in zip(edge_x.tolist(), edge_y.tolist()):
            draw.point((x, y), fill=(255, 230, 0))
    return annotate_image(base, title, subtitle, args.panel_width, args.panel_height)


def make_detection_panel(
    row: pd.Series,
    experiment: str,
    volume: np.ndarray | None,
    mask: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[Image.Image, str]:
    prefix = safe_column(experiment)
    title = f"{EXPERIMENT_LABELS.get(experiment, experiment)} det"
    image_path = Path(str(row[f"{prefix}_image_path"]))
    detector_csv = next(image_path.parent.glob("detector_top*.csv"), None) if image_path.parent.exists() else None
    if volume is None or mask is None:
        subtitle = "missing CT"
        return missing_panel(title, subtitle, args.panel_width, args.panel_height), subtitle
    if detector_csv is None:
        subtitle = "no detector csv"
        return missing_panel(title, subtitle, args.panel_width, args.panel_height), subtitle

    z, slice_2d, mask_slice = full_ct_slice(volume, mask)
    base = Image.fromarray(window_to_uint8(slice_2d, args.window_center, args.window_width), mode="L").convert("RGB")
    draw = ImageDraw.Draw(base)
    if mask_slice.any():
        draw_mask_outline(draw, mask_slice, fill=(255, 230, 0))

    candidates = pd.read_csv(detector_csv)
    in_slice = 0
    for rank, (_, candidate) in enumerate(candidates.iterrows(), start=1):
        cz = float(candidate["coordZ"])
        cy = float(candidate["coordY"])
        cx = float(candidate["coordX"])
        radius = max(3.0, float(candidate.get("radius", 3.0)))
        probability = float(candidate.get("probability", 0.0))
        dz = abs(cz - z)
        color = (0, 255, 255) if dz <= radius + args.hit_margin_voxels else (255, 165, 0)
        if dz <= radius + args.hit_margin_voxels:
            in_slice += 1
        box = (cx - radius, cy - radius, cx + radius, cy + radius)
        for offset in range(2):
            draw.ellipse(
                (box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset),
                outline=color,
            )
        draw.text((cx + radius + 2, cy - radius), f"{rank}:{probability:.2f}", fill=color, font=ImageFont.load_default())

    subtitle = f"{len(candidates)} det | {in_slice} on slice z={z}"
    return annotate_image(base, title, subtitle, args.panel_width, args.panel_height), subtitle


def draw_mask_outline(draw: ImageDraw.ImageDraw, mask_slice: np.ndarray, fill: tuple[int, int, int]) -> None:
    edge = np.logical_xor(mask_slice, ndi.binary_erosion(mask_slice))
    edge_y, edge_x = np.where(edge)
    for x, y in zip(edge_x.tolist(), edge_y.tolist()):
        draw.point((x, y), fill=fill)


def full_ct_slice(volume: np.ndarray, mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    if mask.any():
        zyx = np.argwhere(mask)
        z_values, counts = np.unique(zyx[:, 0], return_counts=True)
        z = int(z_values[np.argmax(counts)])
    else:
        z = volume.shape[0] // 2
    return z, volume[z], mask[z]


def crop_nodule(volume: np.ndarray, mask: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if not mask.any():
        z = volume.shape[0] // 2
        cy = volume.shape[1] // 2
        cx = volume.shape[2] // 2
        half = args.ct_crop_size // 2
    else:
        labeled, n_components = ndi.label(mask)
        if n_components > 1:
            sizes = ndi.sum(mask, labeled, index=np.arange(1, n_components + 1))
            component = labeled == int(np.argmax(sizes) + 1)
        else:
            component = mask
        zyx = np.argwhere(component)
        z_values, counts = np.unique(zyx[:, 0], return_counts=True)
        z = int(z_values[np.argmax(counts)])
        yy, xx = np.where(component[z])
        cy = int(round((int(yy.min()) + int(yy.max())) / 2))
        cx = int(round((int(xx.min()) + int(xx.max())) / 2))
        half = max(args.ct_crop_size // 2, max(int(yy.max() - yy.min()), int(xx.max() - xx.min())) // 2 + args.ct_margin)
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    return (
        crop_with_padding(volume[z], y0, y1, x0, x1, fill=-1000),
        crop_with_padding(mask[z].astype(np.uint8), y0, y1, x0, x1, fill=0).astype(bool),
    )


def make_experiment_panel(
    row: pd.Series,
    experiment: str,
    hit_info: dict[str, object],
    args: argparse.Namespace,
) -> tuple[Image.Image, str]:
    prefix = safe_column(experiment)
    overlay_path = Path(str(row[f"{prefix}_overlay_path"]))
    title = EXPERIMENT_LABELS.get(experiment, experiment)
    correct = bool(row[f"{prefix}_correct"])
    pred = row[f"{prefix}_prediction_name"]
    class_score = float(row[f"{prefix}_true_class_score"])
    hit_label = nodule_hit_label(Path(str(row[f"{prefix}_image_path"])), hit_info, args.hit_margin_voxels)
    subtitle = f"{'OK' if correct else 'ERR'} pred {pred} | cls {class_score:.3f} | {hit_label}"
    if not overlay_path.exists():
        return missing_panel(title, subtitle, args.panel_width, args.panel_height), hit_label
    return annotate_image(Image.open(overlay_path).convert("RGB"), title, subtitle, args.panel_width, args.panel_height), hit_label


def mask_hit_info(mask: np.ndarray | None) -> dict[str, object]:
    if mask is None:
        return {"available": False, "has_mask": False}
    if not mask.any():
        return {"available": True, "has_mask": False}
    zyx = np.argwhere(mask)
    z_min, y_min, x_min = zyx.min(axis=0)
    z_max, y_max, x_max = zyx.max(axis=0)
    return {
        "available": True,
        "has_mask": True,
        "shape": mask.shape,
        "bbox": (int(z_min), int(y_min), int(x_min), int(z_max), int(y_max), int(x_max)),
        "mask": mask,
    }


def nodule_hit_label(image_path: Path, hit_info: dict[str, object], margin: float) -> str:
    if not bool(hit_info["available"]):
        return "hit ?"
    detector_csv = next(image_path.parent.glob("detector_top*.csv"), None) if image_path.parent.exists() else None
    if detector_csv is None:
        return "GT mask"
    candidates = pd.read_csv(detector_csv)
    if candidates.empty:
        return "miss"
    if not bool(hit_info["has_mask"]):
        return "no mask"
    mask = hit_info["mask"]
    shape = hit_info["shape"]
    z_min, y_min, x_min, z_max, y_max, x_max = hit_info["bbox"]
    best_distance = float("inf")
    for _, candidate in candidates.iterrows():
        z = int(round(float(candidate["coordZ"])))
        y = int(round(float(candidate["coordY"])))
        x = int(round(float(candidate["coordX"])))
        if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
            continue
        pad = int(math.ceil(margin))
        z0, z1 = max(0, z - pad), min(shape[0], z + pad + 1)
        y0, y1 = max(0, y - pad), min(shape[1], y + pad + 1)
        x0, x1 = max(0, x - pad), min(shape[2], x + pad + 1)
        if mask[z0:z1, y0:y1, x0:x1].any():
            return "hit"
        dz = axis_distance(z, int(z_min), int(z_max))
        dy = axis_distance(y, int(y_min), int(y_max))
        dx = axis_distance(x, int(x_min), int(x_max))
        best_distance = min(best_distance, math.sqrt(dz * dz + dy * dy + dx * dx))
    if best_distance <= margin:
        return f"hit d~{best_distance:.1f}"
    if best_distance < float("inf"):
        return f"miss d~{best_distance:.1f}"
    return "miss out"


def axis_distance(value: int, low: int, high: int) -> int:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0


def annotate_image(image: Image.Image, title: str, subtitle: str, panel_width: int, panel_height: int) -> Image.Image:
    footer_h = 44
    canvas = Image.new("RGB", (panel_width, panel_height), (18, 18, 18))
    image_area_h = panel_height - footer_h
    image.thumbnail((panel_width, image_area_h), Image.Resampling.LANCZOS)
    x = (panel_width - image.width) // 2
    y = (image_area_h - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, image_area_h, panel_width, panel_height), fill=(18, 18, 18))
    draw.text((6, image_area_h + 5), title[:34], fill=(255, 255, 255), font=ImageFont.load_default())
    draw.text((6, image_area_h + 23), subtitle[:42], fill=(255, 255, 255), font=ImageFont.load_default())
    return canvas


def missing_panel(title: str, subtitle: str, panel_width: int, panel_height: int) -> Image.Image:
    panel = Image.new("RGB", (panel_width, panel_height), (250, 250, 250))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_width - 1, panel_height - 1), outline=(180, 180, 180))
    draw.text((8, 8), "missing", fill=(150, 0, 0), font=ImageFont.load_default())
    return annotate_image(panel, title, subtitle, panel_width, panel_height)


def crop_with_padding(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int, fill: float = 0) -> np.ndarray:
    h, w = arr.shape[:2]
    out = np.full((y1 - y0, x1 - x0), fill, dtype=arr.dtype)
    src_y0, src_y1 = max(y0, 0), min(y1, h)
    src_x0, src_x1 = max(x0, 0), min(x1, w)
    dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0
    out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def window_to_uint8(slice_2d: np.ndarray, center: float, width: float) -> np.ndarray:
    if float(np.nanmin(slice_2d)) >= 0.0 and float(np.nanmax(slice_2d)) <= 255.0:
        return np.clip(slice_2d, 0, 255).astype(np.uint8)
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(slice_2d, low, high)
    return ((clipped - low) / (high - low) * 255.0).astype(np.uint8)


def write_report(report_path: Path, index: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# LUNA16 Synthetic 2D GradCAM Comparison Mosaics",
        "",
        f"- Index: `{output_dir / 'comparison_mosaic_index.csv'}`",
        f"- Rows: `{output_dir / 'comparison_mosaic_rows.csv'}`",
        f"- JPEG per campione: `{output_dir / 'sample_jpegs'}`",
        "",
        "## Come leggere il report",
        "",
        "Questo report confronta, per gli stessi campioni del test set, la CT originale, la sintetica basata su "
        "ground truth e le sintetiche generate a partire dai candidati del detector. Ogni riga del mosaico "
        "corrisponde a un singolo `sample_id`; le colonne mostrano viste diverse dello stesso caso.",
        "",
        "Le prime due colonne sono di riferimento anatomico: `CT slice` mostra la slice CT completa scelta sul "
        "piano in cui la maschera del nodulo ha area massima, senza overlay; `CT + nodule` mostra la stessa "
        "slice con la maschera GT del nodulo sovrapposta in rosso e contorno giallo. Queste due colonne servono "
        "a capire dove si trova realmente il nodulo nella CT.",
        "",
        "Dopo `Synth GT`, ogni esperimento detector-based e' mostrato con due colonne: una colonna `det`, che "
        "disegna sulla CT i candidati predetti dal detector, e una colonna Grad-CAM, che mostra dove il "
        "classificatore guarda sull'immagine sintetica corrispondente. I cerchi cyan indicano candidate vicini "
        "alla slice visualizzata; i cerchi arancioni sono candidate proiettati da slice diverse. La maschera GT "
        "del nodulo, quando presente, e' tracciata in giallo nella colonna detection.",
        "",
        "Le Grad-CAM sono calcolate rispetto alla classe predetta dal modello, quindi spiegano la decisione "
        "effettiva del classificatore. Nei casi sbagliati mostrano quali regioni hanno sostenuto la predizione "
        "errata.",
        "",
        "Sotto ogni pannello sintetico sono riportati: `OK` o `ERR`, cioe se la predizione coincide con la label; "
        "`pred`, cioe la classe predetta; `cls`, cioe la probabilita assegnata alla classe vera; e infine un "
        "indicatore di cattura del nodulo. `hit` significa che almeno un candidato del detector cade vicino alla "
        "maschera GT del nodulo; `miss d~...` indica una mancata cattura con distanza approssimata dal bounding "
        "box della maschera; `no mask` indica un caso benigno senza maschera nodulo; `GT mask` indica una colonna "
        "non basata su detector, come la sintetica ground truth.",
        "",
        "Oltre alle pagine aggregate, la cartella `sample_jpegs/` contiene un JPEG per ogni singolo campione e "
        "categoria. Questi file sono piu comodi quando si vuole zoomare su un caso specifico senza aprire una "
        "pagina mosaico intera.",
        "",
        "Le sezioni del report separano i casi in categorie diagnostiche. `gt_correct_top5_wrong` e' la categoria "
        "piu utile per capire perche la sintetica Top5 perde casi che la sintetica GT classifica bene. "
        "`gt_wrong_top5_correct` mostra invece i casi in cui il detector aiuta. `top5_false_positive` e "
        "`top5_false_negative` isolano gli errori Top5 per classe.",
        "",
    ]
    for category in sorted(index["category"].unique()) if not index.empty else []:
        lines.extend(["", f"## {category}", ""])
        for _, row in index[index["category"] == category].iterrows():
            rel_path = Path(str(row["path"])).relative_to(output_dir).as_posix()
            lines.append(f'<img src="{rel_path}" width="2000">')
            lines.append("")
    report_path.write_text("\n".join(lines).rstrip() + "\n")


def safe_column(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def prefer_existing_path(primary: object, fallback: object) -> Path:
    primary_path = Path(str(primary)) if not pd.isna(primary) and str(primary) else None
    fallback_path = Path(str(fallback))
    if primary_path is not None and primary_path.exists():
        return primary_path
    return fallback_path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def short_id(value: object) -> str:
    text = str(value)
    return text[-18:] if len(text) > 18 else text


if __name__ == "__main__":
    main()
