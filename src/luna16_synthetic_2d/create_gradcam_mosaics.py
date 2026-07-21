from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MANIFEST = Path("docs/luna16_synthetic_2d_gradcam_all_predicted/gradcam_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("docs/luna16_synthetic_2d_gradcam_mosaics")
DEFAULT_REFERENCE_EXPERIMENT = "luna16_synthetic_2d_gt"
DEFAULT_COMPARE_EXPERIMENTS = (
    "luna16_synthetic_2d_top5_minprob0.5",
    "luna16_synthetic_2d_top7_minprob0.3_rbf",
    "luna16_synthetic_2d_top7_minprob0.3_shepard",
    "luna16_synthetic_2d_top4_minprob0.5_rbf",
    "luna16_synthetic_2d_top4_minprob0.5_shepard",
    "luna16_synthetic_2d_top3_minprob0.5_rbf",
    "luna16_synthetic_2d_top3_minprob0.5_shepard",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paired GradCAM mosaics from a GradCAM manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-experiment", default=DEFAULT_REFERENCE_EXPERIMENT)
    parser.add_argument("--compare-experiments", nargs="+", default=list(DEFAULT_COMPARE_EXPERIMENTS))
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument("--max-samples-per-category", type=int, default=48)
    parser.add_argument("--rows-per-page", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=384)
    parser.add_argument("--tile-height", type=int, default=350)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[(manifest["gradcam_status"] == "ok") & (manifest["backbone"] == args.backbone)].copy()
    if manifest.empty:
        raise RuntimeError(f"No GradCAM rows found in {args.manifest} for backbone {args.backbone}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, object]] = []

    for compare_experiment in args.compare_experiments:
        pairs = build_pairs(manifest, args.reference_experiment, compare_experiment)
        if pairs.empty:
            continue
        for category, category_rows in category_frames(pairs).items():
            if category_rows.empty:
                continue
            selected = category_rows.head(args.max_samples_per_category).copy()
            pages.extend(
                save_pages(
                    selected,
                    args.output_dir / safe_name(compare_experiment),
                    compare_experiment,
                    category,
                    args.rows_per_page,
                    args.tile_width,
                    args.tile_height,
                )
            )

    index = pd.DataFrame(pages)
    index.to_csv(args.output_dir / "gradcam_mosaic_index.csv", index=False)
    write_report(args.output_dir / "gradcam_mosaic_report.md", index, args.output_dir)
    print(f"Wrote {len(index)} GradCAM mosaic pages to {args.output_dir}")


def build_pairs(manifest: pd.DataFrame, reference_experiment: str, compare_experiment: str) -> pd.DataFrame:
    key = ["sample_id", "label", "label_name"]
    reference = manifest[manifest["experiment"] == reference_experiment].copy()
    compare = manifest[manifest["experiment"] == compare_experiment].copy()
    merged = reference.merge(compare, on=key, suffixes=("_ref", "_cmp"))
    if merged.empty:
        return merged
    merged["score_gap"] = (
        merged["true_class_score_ref"].astype(float) - merged["true_class_score_cmp"].astype(float)
    ).abs()
    merged["compare_error_margin"] = merged["error_margin_cmp"].astype(float).abs()
    return merged.sort_values(["score_gap", "compare_error_margin"], ascending=False)


def category_frames(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    gt_ok = pairs["correct_ref"].astype(bool)
    cmp_ok = pairs["correct_cmp"].astype(bool)
    benign = pairs["label_name"] == "benign"
    malignant = pairs["label_name"] == "malignant"
    cmp_pred_malignant = pairs["prediction_name_cmp"] == "malignant"
    cmp_pred_benign = pairs["prediction_name_cmp"] == "benign"
    return {
        "gt_correct_synthetic_correct": pairs[gt_ok & cmp_ok],
        "gt_correct_synthetic_wrong": pairs[gt_ok & ~cmp_ok],
        "gt_wrong_synthetic_correct": pairs[~gt_ok & cmp_ok],
        "both_wrong": pairs[~gt_ok & ~cmp_ok],
        "synthetic_false_positive": pairs[benign & cmp_pred_malignant],
        "synthetic_false_negative": pairs[malignant & cmp_pred_benign],
    }


def save_pages(
    rows: pd.DataFrame,
    output_dir: Path,
    compare_experiment: str,
    category: str,
    rows_per_page: int,
    tile_width: int,
    tile_height: int,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    page_count = math.ceil(len(rows) / rows_per_page)
    page_rows: list[dict[str, object]] = []
    for page_idx in range(page_count):
        page = rows.iloc[page_idx * rows_per_page : (page_idx + 1) * rows_per_page]
        canvas = Image.new("RGB", (tile_width * 2, tile_height * len(page)), (245, 245, 245))
        for row_idx, (_, row) in enumerate(page.iterrows()):
            y = row_idx * tile_height
            canvas.paste(
                make_tile(Path(str(row["gradcam_overlay_path_ref"])), ref_title(row), tile_width, tile_height),
                (0, y),
            )
            canvas.paste(
                make_tile(Path(str(row["gradcam_overlay_path_cmp"])), cmp_title(row), tile_width, tile_height),
                (tile_width, y),
            )
        page_path = output_dir / f"{safe_name(category)}_page_{page_idx + 1:03d}.jpg"
        canvas.save(page_path, quality=92)
        page_rows.append(
            {
                "compare_experiment": compare_experiment,
                "category": category,
                "page": page_idx + 1,
                "samples": len(page),
                "path": str(page_path),
            }
        )
    return page_rows


def make_tile(image_path: Path, title: str, tile_width: int, tile_height: int) -> Image.Image:
    tile = Image.new("RGB", (tile_width, tile_height), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    image_area_h = tile_height - 58
    if image_path.exists():
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((tile_width, image_area_h), Image.Resampling.LANCZOS)
        x = (tile_width - image.width) // 2
        y = (image_area_h - image.height) // 2
        tile.paste(image, (x, y))
    else:
        draw.rectangle((0, 0, tile_width - 1, image_area_h - 1), outline=(170, 170, 170))
        draw.text((8, 8), "missing image", fill=(120, 0, 0))
    draw.rectangle((0, image_area_h, tile_width, tile_height), fill=(24, 24, 24))
    draw.multiline_text((8, image_area_h + 6), wrap_text(title, 38), fill=(255, 255, 255), font=ImageFont.load_default())
    return tile


def ref_title(row: pd.Series) -> str:
    return (
        f"GT | true {row['label_name']} | pred {row['prediction_name_ref']}\n"
        f"class {float(row['true_class_score_ref']):.3f} | sample {short_id(row['sample_id'])}"
    )


def cmp_title(row: pd.Series) -> str:
    return (
        f"SYN | true {row['label_name']} | pred {row['prediction_name_cmp']}\n"
        f"class {float(row['true_class_score_cmp']):.3f} | gap {float(row['score_gap']):.3f}"
    )


def write_report(report_path: Path, index: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# LUNA16 Synthetic 2D GradCAM Mosaics",
        "",
        f"- Index: `{output_dir / 'gradcam_mosaic_index.csv'}`",
        "",
    ]
    if index.empty:
        lines.append("No mosaic pages were generated.")
    else:
        for compare_experiment in sorted(index["compare_experiment"].unique()):
            lines.extend(["", f"## {compare_experiment}", ""])
            exp_rows = index[index["compare_experiment"] == compare_experiment]
            for category in sorted(exp_rows["category"].unique()):
                lines.extend(["", f"### {category}", ""])
                for _, row in exp_rows[exp_rows["category"] == category].iterrows():
                    rel_path = Path(str(row["path"])).relative_to(output_dir).as_posix()
                    lines.append(f'<img src="{rel_path}" width="900">')
                    lines.append("")
    report_path.write_text("\n".join(lines).rstrip() + "\n")


def wrap_text(value: str, width: int) -> str:
    lines: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line
        while len(line) > width:
            lines.append(line[:width])
            line = line[width:]
        lines.append(line)
    return "\n".join(lines)


def short_id(value: object) -> str:
    text = str(value)
    return text[-18:] if len(text) > 18 else text


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    main()
