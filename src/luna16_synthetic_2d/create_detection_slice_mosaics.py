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


DEFAULT_ROWS_CSV = Path(
    "docs/luna16_synthetic_2d_gradcam_comparison_mosaics_predicted_detection_jpegs/comparison_mosaic_rows.csv"
)
FALLBACK_ROWS_CSV = Path("docs/luna16_synthetic_2d_detection_slice_mosaics/detection_slice_mosaic_index.csv")
DEFAULT_GRADCAM_MANIFEST = Path("docs/luna16_synthetic_2d_gradcam_all_predicted/gradcam_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("docs/luna16_synthetic_2d_detection_slice_mosaics")
SYNTH_GT_EXPERIMENT = "luna16_synthetic_2d_gt"
DETECTOR_EXPERIMENTS = (
    ("Top3 RBF", "luna16_synthetic_2d_top3_minprob0.5_rbf", "luna16_saliency_synthetic_detector_top3_minprob0.5_rbf"),
    ("Top4 RBF", "luna16_synthetic_2d_top4_minprob0.5_rbf", "luna16_saliency_synthetic_detector_top4_minprob0.5_rbf"),
    ("Top5", "luna16_synthetic_2d_top5_minprob0.5", "luna16_saliency_synthetic_detector_top5_minprob0.5"),
    ("Top7 RBF", "luna16_synthetic_2d_top7_minprob0.3_rbf", "luna16_saliency_synthetic_detector_top7_minprob0.3_rbf"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-sample mosaics showing each detector candidate on its CT slice.")
    parser.add_argument("--rows-csv", type=Path, default=DEFAULT_ROWS_CSV)
    parser.add_argument("--gradcam-manifest", type=Path, default=DEFAULT_GRADCAM_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-source",
        choices=["all", "categories", "classified"],
        default="all",
        help=(
            "Use all test samples, the focused category rows CSV, or split all samples by "
            "GT-vs-Top5 classification outcome."
        ),
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/LUNA16_preprocessed"))
    parser.add_argument("--synthetic-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples for previews; 0 means all rows.")
    parser.add_argument("--tile-size", type=int, default=176)
    parser.add_argument("--label-width", type=int, default=150)
    parser.add_argument("--heatmap-width", type=int, default=220)
    parser.add_argument("--max-detections", type=int, default=7)
    parser.add_argument("--window-center", type=float, default=-600.0)
    parser.add_argument("--window-width", type=float, default=1500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.processed_dir.exists():
        raise FileNotFoundError(f"Processed LUNA16 directory does not exist: {args.processed_dir}")
    if not args.synthetic_dir.exists():
        raise FileNotFoundError(f"Synthetic LUNA16 directory does not exist: {args.synthetic_dir}")
    gradcam_frame = pd.read_csv(args.gradcam_manifest)
    rows = load_sample_rows(args, gradcam_frame)
    gradcam = load_gradcam_manifest(gradcam_frame)
    if args.max_samples > 0:
        rows = rows.head(args.max_samples).copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    volume_cache: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
    records: list[dict[str, object]] = []
    category_counts: dict[str, int] = {}
    for row_index, row in rows.iterrows():
        sample_id = str(row["sample_id"])
        category = str(row["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
        volume, mask = load_volume_mask(sample_id, args.processed_dir, volume_cache)
        out_dir = args.output_dir / "sample_jpegs" / safe_name(category)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_name(category)}_{category_counts[category]:04d}.jpg"
        canvas, summary = make_sample_mosaic(sample_id, str(row["label_name"]), volume, mask, gradcam, args)
        canvas.save(out_path, quality=92)
        record = {
            "category": category,
            "sample_id": sample_id,
            "label_name": row["label_name"],
            "path": str(out_path),
        }
        record.update(summary)
        records.append(record)

    index = pd.DataFrame(records)
    index.to_csv(args.output_dir / "detection_slice_mosaic_index.csv", index=False)
    write_report(args.output_dir / "detection_slice_mosaic_report.md", index, args.output_dir)
    print(f"Wrote {len(index)} detection-slice mosaics to {args.output_dir}")


def load_sample_rows(args: argparse.Namespace, gradcam_frame: pd.DataFrame) -> pd.DataFrame:
    if args.sample_source == "categories":
        rows_csv = args.rows_csv if args.rows_csv.exists() else FALLBACK_ROWS_CSV
        return pd.read_csv(rows_csv)
    if args.sample_source == "classified":
        return build_classified_sample_rows(gradcam_frame)

    frame = gradcam_frame[
        (gradcam_frame["gradcam_status"] == "ok")
        & (gradcam_frame["experiment"] == SYNTH_GT_EXPERIMENT)
    ].copy()
    frame = frame.sort_values(["label_name", "rank_within_true_class", "sample_id"])
    rows = frame[["sample_id", "label_name"]].drop_duplicates("sample_id").copy()
    rows["category"] = "all_samples"
    return rows[["category", "sample_id", "label_name"]].reset_index(drop=True)


def build_classified_sample_rows(gradcam_frame: pd.DataFrame) -> pd.DataFrame:
    frame = gradcam_frame[gradcam_frame["gradcam_status"] == "ok"].copy()
    gt = frame[frame["experiment"] == SYNTH_GT_EXPERIMENT].copy()
    top5 = frame[frame["experiment"] == "luna16_synthetic_2d_top5_minprob0.5"].copy()
    merged = gt.merge(
        top5,
        on=["sample_id", "label", "label_name"],
        suffixes=("_gt", "_top5"),
    )
    records: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        gt_correct = bool(row["correct_gt"])
        top5_correct = bool(row["correct_top5"])
        label = int(row["label"])
        top5_prediction = int(row["prediction_top5"])
        categories = []
        if gt_correct and top5_correct:
            categories.append("both_correct")
        elif (not gt_correct) and (not top5_correct):
            categories.append("both_wrong")
        elif gt_correct and not top5_correct:
            categories.append("gt_correct_top5_wrong")
        elif (not gt_correct) and top5_correct:
            categories.append("gt_wrong_top5_correct")
        if label == 0 and top5_prediction == 1:
            categories.append("top5_false_positive")
        if label == 1 and top5_prediction == 1:
            categories.append("top5_true_positive")
        for category in categories:
            records.append(
                {
                    "category": category,
                    "sample_id": row["sample_id"],
                    "label_name": row["label_name"],
                    "sort_score": abs(float(row["true_class_score_gt"]) - float(row["true_class_score_top5"])),
                }
            )
    result = pd.DataFrame(records)
    if result.empty:
        return result
    return result.sort_values(["category", "sort_score", "sample_id"], ascending=[True, False, True]).reset_index(drop=True)


def load_gradcam_manifest(frame: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    frame = frame[frame["gradcam_status"] == "ok"].copy()
    result: dict[tuple[str, str], pd.Series] = {}
    for _, row in frame.iterrows():
        result[(str(row["sample_id"]), str(row["experiment"]))] = row
    return result


def load_volume_mask(
    sample_id: str,
    processed_dir: Path,
    cache: dict[str, tuple[np.ndarray | None, np.ndarray | None]],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if sample_id in cache:
        return cache[sample_id]
    volume_path = resolve_processed_file(processed_dir, sample_id, "volume")
    mask_path = resolve_processed_file(processed_dir, sample_id, "nodule_mask")
    if not volume_path.exists() or not mask_path.exists():
        cache[sample_id] = (None, None)
        return cache[sample_id]
    volume = nib.load(str(volume_path)).get_fdata().astype(np.float32).transpose(2, 1, 0)
    mask = nib.load(str(mask_path)).get_fdata().astype(np.float32).transpose(2, 1, 0) > 0
    cache[sample_id] = (volume, mask)
    return cache[sample_id]


def resolve_processed_file(processed_dir: Path, sample_id: str, suffix: str) -> Path:
    direct = processed_dir / sample_id / f"{sample_id}_{suffix}.nii.gz"
    if direct.exists():
        return direct
    for subset_idx in range(10):
        candidate = processed_dir / f"subset{subset_idx}" / sample_id / f"{sample_id}_{suffix}.nii.gz"
        if candidate.exists():
            return candidate
    return direct


def make_sample_mosaic(
    sample_id: str,
    label_name: str,
    volume: np.ndarray | None,
    mask: np.ndarray | None,
    gradcam: dict[tuple[str, str], pd.Series],
    args: argparse.Namespace,
) -> tuple[Image.Image, dict[str, object]]:
    rows_count = 1 + len(DETECTOR_EXPERIMENTS)
    gt_tiles = make_gt_tiles(sample_id, volume, mask, args) if volume is not None and mask is not None else []
    tile_columns = max(args.max_detections, len(gt_tiles), 1)
    width = args.label_width + tile_columns * args.tile_size + args.heatmap_width
    height = rows_count * args.tile_size
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    summary: dict[str, object] = {}

    if volume is None or mask is None:
        draw_missing(canvas, f"{sample_id}\nmissing CT/mask")
        return canvas, summary

    canvas.paste(make_label_panel("GT nodules", f"{len(gt_tiles)} | true {label_name}", args), (0, 0))
    for gt_idx, gt_panel in enumerate(gt_tiles):
        canvas.paste(gt_panel, (args.label_width + gt_idx * args.tile_size, 0))
    synth_gt_panel = make_gradcam_panel(gradcam.get((sample_id, SYNTH_GT_EXPERIMENT)), "Synth GT", args)
    canvas.paste(synth_gt_panel, (args.label_width + tile_columns * args.tile_size, 0))
    summary["gt_nodule_count"] = len(gt_tiles)

    for row_idx, (label, experiment, root_name) in enumerate(DETECTOR_EXPERIMENTS, start=1):
        y = row_idx * args.tile_size
        root = args.synthetic_dir / root_name
        detector_csv = detector_csv_path(root, sample_id)
        if detector_csv is None:
            canvas.paste(make_label_panel(label, "no csv", args), (0, y))
            gradcam_panel = make_gradcam_panel(gradcam.get((sample_id, experiment)), label, args)
            canvas.paste(gradcam_panel, (args.label_width + tile_columns * args.tile_size, y))
            summary[f"{safe_column(label)}_count"] = 0
            summary[f"{safe_column(label)}_hits"] = ""
            continue
        candidates = pd.read_csv(detector_csv).head(args.max_detections)
        canvas.paste(make_label_panel(label, f"{len(candidates)} detections", args), (0, y))
        hits = 0
        for det_idx, (_, candidate) in enumerate(candidates.iterrows()):
            tile, is_hit = make_detection_tile(volume, mask, candidate, det_idx + 1, args)
            if is_hit:
                hits += 1
            canvas.paste(tile, (args.label_width + det_idx * args.tile_size, y))
        gradcam_panel = make_gradcam_panel(gradcam.get((sample_id, experiment)), label, args)
        canvas.paste(gradcam_panel, (args.label_width + tile_columns * args.tile_size, y))
        summary[f"{safe_column(label)}_count"] = len(candidates)
        summary[f"{safe_column(label)}_hits"] = hits
    return canvas, summary


def make_gradcam_panel(row: pd.Series | None, label: str, args: argparse.Namespace) -> Image.Image:
    if row is None:
        return make_missing_gradcam(label, "missing Grad-CAM", args)
    overlay_path = Path(str(row["gradcam_overlay_path"]))
    if not overlay_path.exists():
        return make_missing_gradcam(label, "missing image", args)
    image = Image.open(overlay_path).convert("RGB")
    correct = bool(row["correct"])
    pred = str(row["prediction_name"])
    true_class_score = float(row["true_class_score"])
    title = f"{label} Grad-CAM"
    subtitle = f"{'OK' if correct else 'ERR'} pred {pred} | cls {true_class_score:.3f}"
    return annotate_rect(image, title, subtitle, args.heatmap_width, args.tile_size)


def make_missing_gradcam(title: str, subtitle: str, args: argparse.Namespace) -> Image.Image:
    panel = Image.new("RGB", (args.heatmap_width, args.tile_size), (28, 28, 28))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), title, fill=(255, 255, 255), font=ImageFont.load_default())
    draw.text((8, 28), subtitle, fill=(255, 130, 130), font=ImageFont.load_default())
    return panel


def make_gt_tiles(sample_id: str, volume: np.ndarray, mask: np.ndarray, args: argparse.Namespace) -> list[Image.Image]:
    if not mask.any():
        z = volume.shape[0] // 2
        image = ct_rgb(volume[z], args)
        return [annotate(image, f"GT z={z}", short_id(sample_id), args.tile_size)]

    labeled, n_components = ndi.label(mask)
    tiles: list[Image.Image] = []
    for component_idx in range(1, n_components + 1):
        component = labeled == component_idx
        if not component.any():
            continue
        z = nodule_slice(component)
        image = ct_rgb(volume[z], args)
        overlay_mask(image, mask[z], alpha=0.30)
        draw = ImageDraw.Draw(image)
        draw_mask_outline(draw, mask[z], (255, 230, 0))
        draw_mask_outline(draw, component[z], (0, 255, 255))
        tiles.append(annotate(image, f"GT #{component_idx} z={z}", short_id(sample_id), args.tile_size))
    return tiles


def make_detection_tile(
    volume: np.ndarray,
    mask: np.ndarray,
    candidate: pd.Series,
    rank: int,
    args: argparse.Namespace,
) -> tuple[Image.Image, bool]:
    z = int(round(float(candidate["coordZ"])))
    y = float(candidate["coordY"])
    x = float(candidate["coordX"])
    radius = max(3.0, float(candidate.get("radius", 3.0)))
    probability = float(candidate.get("probability", 0.0))
    z = max(0, min(volume.shape[0] - 1, z))
    image = ct_rgb(volume[z], args)
    draw = ImageDraw.Draw(image)
    if mask[z].any():
        draw_mask_outline(draw, mask[z], (255, 230, 0))
    is_hit = detection_hits_mask(mask, z, y, x, radius)
    color = (0, 255, 255) if is_hit else (255, 165, 0)
    draw_detection(draw, x, y, radius, color)
    return annotate(image, f"#{rank} z={z} p={probability:.2f}", "hit" if is_hit else "miss", args.tile_size), is_hit


def detection_hits_mask(mask: np.ndarray, z: int, y: float, x: float, radius: float) -> bool:
    if not mask.any():
        return False
    pad = int(math.ceil(radius + 8))
    zi = int(round(z))
    yi = int(round(y))
    xi = int(round(x))
    z0, z1 = max(0, zi - pad), min(mask.shape[0], zi + pad + 1)
    y0, y1 = max(0, yi - pad), min(mask.shape[1], yi + pad + 1)
    x0, x1 = max(0, xi - pad), min(mask.shape[2], xi + pad + 1)
    return bool(mask[z0:z1, y0:y1, x0:x1].any())


def detector_csv_path(root: Path, sample_id: str) -> Path | None:
    sample_dir = root / sample_id
    if not sample_dir.exists():
        return None
    files = sorted(sample_dir.glob("detector_top*.csv"))
    return files[0] if files else None


def nodule_slice(mask: np.ndarray) -> int:
    zyx = np.argwhere(mask)
    z_values, counts = np.unique(zyx[:, 0], return_counts=True)
    return int(z_values[np.argmax(counts)])


def ct_rgb(slice_2d: np.ndarray, args: argparse.Namespace) -> Image.Image:
    return Image.fromarray(window_to_uint8(slice_2d, args.window_center, args.window_width), mode="L").convert("RGB")


def window_to_uint8(slice_2d: np.ndarray, center: float, width: float) -> np.ndarray:
    if float(np.nanmin(slice_2d)) >= 0.0 and float(np.nanmax(slice_2d)) <= 255.0:
        return np.clip(slice_2d, 0, 255).astype(np.uint8)
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(slice_2d, low, high)
    return ((clipped - low) / (high - low) * 255.0).astype(np.uint8)


def overlay_mask(image: Image.Image, mask_slice: np.ndarray, alpha: float) -> None:
    mask_img = Image.fromarray((mask_slice.astype(np.uint8) * 255), mode="L")
    red = Image.new("RGB", image.size, (255, 20, 20))
    image.paste(Image.composite(Image.blend(image, red, alpha), image, mask_img))


def draw_mask_outline(draw: ImageDraw.ImageDraw, mask_slice: np.ndarray, fill: tuple[int, int, int]) -> None:
    edge = np.logical_xor(mask_slice, ndi.binary_erosion(mask_slice))
    edge_y, edge_x = np.where(edge)
    for x, y in zip(edge_x.tolist(), edge_y.tolist()):
        draw.point((x, y), fill=fill)


def draw_detection(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float, color: tuple[int, int, int]) -> None:
    box = (x - radius, y - radius, x + radius, y + radius)
    for offset in range(2):
        draw.ellipse((box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset), outline=color)
    draw.line((x - radius, y, x + radius, y), fill=color)
    draw.line((x, y - radius, x, y + radius), fill=color)


def annotate(image: Image.Image, title: str, subtitle: str, tile_size: int) -> Image.Image:
    footer_h = 34
    image = image.copy()
    image.thumbnail((tile_size, tile_size - footer_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile_size, tile_size), (18, 18, 18))
    canvas.paste(image, ((tile_size - image.width) // 2, (tile_size - footer_h - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, tile_size - footer_h, tile_size, tile_size), fill=(18, 18, 18))
    draw.text((5, tile_size - footer_h + 4), title[:28], fill=(255, 255, 255), font=ImageFont.load_default())
    draw.text((5, tile_size - footer_h + 18), subtitle[:28], fill=(255, 255, 255), font=ImageFont.load_default())
    return canvas


def annotate_rect(image: Image.Image, title: str, subtitle: str, width: int, height: int) -> Image.Image:
    footer_h = 34
    image = image.copy()
    image.thumbnail((width, height - footer_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    canvas.paste(image, ((width - image.width) // 2, (height - footer_h - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, height - footer_h, width, height), fill=(18, 18, 18))
    draw.text((5, height - footer_h + 4), title[:32], fill=(255, 255, 255), font=ImageFont.load_default())
    draw.text((5, height - footer_h + 18), subtitle[:40], fill=(255, 255, 255), font=ImageFont.load_default())
    return canvas


def make_label_panel(title: str, subtitle: str, args: argparse.Namespace) -> Image.Image:
    panel = Image.new("RGB", (args.label_width, args.tile_size), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 10), title[:20], fill=(255, 255, 255), font=ImageFont.load_default())
    draw.text((8, 30), subtitle[:22], fill=(210, 210, 210), font=ImageFont.load_default())
    return panel


def draw_missing(canvas: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), text, fill=(255, 120, 120), font=ImageFont.load_default())


def write_report(report_path: Path, index: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# LUNA16 Synthetic 2D Detection Slice Mosaics",
        "",
        f"- Index: `{output_dir / 'detection_slice_mosaic_index.csv'}`",
        f"- JPEG per campione: `{output_dir / 'sample_jpegs'}`",
        "",
        "Ogni immagine mostra, per lo stesso campione, la slice GT del nodulo, le slice CT corrispondenti "
        "a ciascuna detection usata per generare le sintetiche Top3, Top4, Top5 e Top7, e la Grad-CAM "
        "del classificatore per la sintetica corrispondente.",
        "",
        "Nei pannelli delle detection, il cerchio cyan indica che la detection cade vicino alla maschera GT "
        "del nodulo; il cerchio arancione indica una detection che non intercetta il nodulo. Il contorno giallo "
        "mostra la maschera GT quando e' presente su quella slice.",
        "",
        "La colonna finale di ogni riga mostra la Grad-CAM predittiva: `OK/ERR` indica se la classificazione "
        "e' corretta, `pred` e' la classe predetta e `cls` e' lo score della classe vera. In questo modo il "
        "mosaico collega direttamente cio che prende il detector con cio che usa il classificatore.",
        "",
        "Quando il report e' generato in modalita `classified`, i campioni sono divisi confrontando la predizione "
        "del classificatore su `Synth GT` con quella su `Top5 minprob0.5`: `both_wrong`, `both_correct`, "
        "`gt_correct_top5_wrong`, `gt_wrong_top5_correct`, `top5_false_positive` e `top5_true_positive`.",
        "",
    ]
    for category in sorted(index["category"].unique()) if not index.empty else []:
        lines.extend(["", f"## {category}", ""])
        rows = index[index["category"] == category]
        for _, row in rows.iterrows():
            rel = Path(str(row["path"])).relative_to(output_dir).as_posix()
            lines.append(f'<img src="{rel}" width="1400">')
            lines.append("")
    report_path.write_text("\n".join(lines).rstrip() + "\n")


def safe_column(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def short_id(value: object) -> str:
    text = str(value)
    return text[-18:] if len(text) > 18 else text


if __name__ == "__main__":
    main()
