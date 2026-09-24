"""Generate the medical-image assets used in manuscript Figure 2.

The assets are intentionally exported without panel titles or surrounding
method boxes so they can be positioned directly in the PowerPoint source.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_UID = "1.3.6.1.4.1.14519.5.2.1.6279.6001.177086402277715068525592995222"
DETECTION_COLOR = "#E69F00"
GROUND_TRUTH_COLOR = "#D62728"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid", default=DEFAULT_UID)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=Path(
            "data/synthetic_2d/"
            "luna16_saliency_synthetic_detector_cpmnetv2_bf16_top4_minprob0.50_rbf"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("manuscript/figures/figure2_assets")
    )
    return parser.parse_args()


def find_processed_volume(root: Path, uid: str) -> Path:
    direct = root / uid / f"{uid}_volume.nii.gz"
    if direct.exists():
        return direct
    for subset in range(10):
        candidate = root / f"subset{subset}" / uid / f"{uid}_volume.nii.gz"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate preprocessed volume for {uid}")


def load_volume(path: Path) -> np.ndarray:
    volume = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {volume.shape}")
    return volume


def ct_uint8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size and float(finite.min()) >= 0 and float(finite.max()) <= 255:
        return np.clip(image, 0, 255).astype(np.uint8)
    low, high = -1350.0, 150.0  # lung window: center -600, width 1500
    return np.rint((np.clip(image, low, high) - low) * 255.0 / (high - low)).astype(np.uint8)


def rgba_slice(volume: np.ndarray, z: int) -> Image.Image:
    z = int(np.clip(z, 0, volume.shape[0] - 1))
    gray = Image.fromarray(ct_uint8(volume[z]), mode="L")
    return gray.convert("RGBA")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size=size)


def outlined_ellipse(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: str,
    width: int,
) -> None:
    halo = max(2, width // 2)
    draw.ellipse(box, outline="#111111", width=width + 2 * halo)
    draw.ellipse(box, outline="#FFFFFF", width=width + halo)
    draw.ellipse(box, outline=color, width=width)


def label_badge(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: str,
    text_color: str,
    label_font: ImageFont.FreeTypeFont,
    padding: tuple[int, int] = (13, 7),
) -> tuple[int, int, int, int]:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=label_font)
    width = bbox[2] - bbox[0] + 2 * padding[0]
    height = bbox[3] - bbox[1] + 2 * padding[1]
    rect = (x, y, x + width, y + height)
    draw.rounded_rectangle(rect, radius=max(8, height // 4), fill=color, outline="#FFFFFF", width=3)
    draw.text(
        (x + padding[0], y + padding[1] - bbox[1]),
        text,
        font=label_font,
        fill=text_color,
    )
    return rect


def read_detections(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    detections: list[dict[str, float]] = []
    for row in rows[:4]:
        detections.append(
            {
                key: float(row[key])
                for key in ("coordZ", "coordY", "coordX", "radius", "probability")
            }
        )
    if len(detections) != 4:
        raise ValueError(f"Expected four detector candidates, found {len(detections)}")
    return detections


def create_ct_stack(volume: np.ndarray, output: Path) -> None:
    """Create a clean volume stack without labels that could imply GT use."""
    target_w = 760
    source_h, source_w = volume.shape[1:]
    target_h = int(round(target_w * source_h / source_w))
    z_values = [82, 102, 122, 140, 154, 163, 170]
    offset_x, offset_y = 38, 31
    border = 7
    canvas_w = target_w + (len(z_values) - 1) * offset_x + 2 * border
    canvas_h = target_h + (len(z_values) - 1) * offset_y + 2 * border
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))

    # Farther slices are lighter; the foremost slice remains fully opaque.
    for layer, z in enumerate(z_values):
        image = rgba_slice(volume, z).resize((target_w, target_h), Image.Resampling.LANCZOS)
        opacity = int(round(95 + layer * (160 / (len(z_values) - 1))))
        image.putalpha(opacity)
        edge = "#D8DDE3" if layer < len(z_values) - 1 else "#FFFFFF"
        card = ImageOps.expand(image, border=border, fill=edge)
        x = (len(z_values) - 1 - layer) * offset_x
        y = (len(z_values) - 1 - layer) * offset_y
        canvas.alpha_composite(card, (x, y))

    canvas.save(output)


def detection_tile(
    volume: np.ndarray,
    detection: dict[str, float],
    index: int,
    image_width: int = 570,
) -> Image.Image:
    source_h, source_w = volume.shape[1:]
    image_height = int(round(image_width * source_h / source_w))
    z = int(round(detection["coordZ"]))
    image = rgba_slice(volume, z).resize((image_width, image_height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    scale_x = image_width / source_w
    scale_y = image_height / source_h
    x = detection["coordX"] * scale_x
    y = detection["coordY"] * scale_y
    radius = max(13.0, detection["radius"] * (scale_x + scale_y) / 2.0)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=DETECTION_COLOR,
        width=3,
    )
    label_badge(
        draw, (16, 16), f"D{index + 1}   p={detection['probability']:.2f}",
        DETECTION_COLOR, "#111111", font(29, bold=True)
    )
    return ImageOps.expand(image, border=5, fill="#FFFFFF")


def create_detector_grid(
    volume: np.ndarray,
    detections: list[dict[str, float]],
    output: Path,
) -> None:
    tiles = [detection_tile(volume, detection, index) for index, detection in enumerate(detections)]
    tile_w, tile_h = tiles[0].size
    gap = 22
    canvas = Image.new(
        "RGBA", (2 * tile_w + gap, 2 * tile_h + gap),
        (255, 255, 255, 0)
    )
    for index, tile in enumerate(tiles):
        x = (index % 2) * (tile_w + gap)
        y = (index // 2) * (tile_h + gap)
        canvas.alpha_composite(tile, (x, y))
    canvas.save(output)


def create_annotated_synthetic(
    source: Path,
    detections: list[dict[str, float]],
    output: Path,
) -> None:
    scale = 4
    image = Image.open(source).convert("RGBA")
    source_w, source_h = image.size
    image = image.resize((source_w * scale, source_h * scale), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)

    label_offsets = ((42, -88), (35, 26), (-142, -74), (34, 26))
    for index, detection in enumerate(detections):
        color = DETECTION_COLOR
        x = detection["coordX"] * scale
        y = detection["coordY"] * scale
        radius = max(8.0, detection["radius"]) * scale
        outlined_ellipse(draw, (x - radius, y - radius, x + radius, y + radius), color, width=8)
        center_r = 7
        draw.ellipse((x - center_r, y - center_r, x + center_r, y + center_r), fill=color, outline="#FFFFFF", width=3)

        dx, dy = label_offsets[index]
        label_x = int(np.clip(x + dx, 8, image.width - 105))
        label_y = int(np.clip(y + dy, 8, image.height - 65))
        text_color = "#111111"
        rect = label_badge(
            draw,
            (label_x, label_y),
            f"D{index + 1}",
            color,
            text_color,
            font(34, bold=True),
            padding=(13, 7),
        )
        anchor_x = rect[0] if label_x >= x else rect[2]
        anchor_y = (rect[1] + rect[3]) // 2
        draw.line((x, y, anchor_x, anchor_y), fill="#FFFFFF", width=8)
        draw.line((x, y, anchor_x, anchor_y), fill=color, width=4)

    image.save(output)


def main() -> None:
    args = parse_args()
    sample_dir = args.synthetic_root / args.uid
    volume_path = find_processed_volume(args.processed_root, args.uid)
    detector_csv = sample_dir / f"detector_top4_{args.uid}.csv"
    synthetic_png = sample_dir / f"surface_{args.uid}.png"
    for path in (volume_path, detector_csv, synthetic_png):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    volume = load_volume(volume_path)
    detections = read_detections(detector_csv)

    outputs = {
        "ct stack": args.output_dir / "figure2_ct_slice_stack.png",
        "detector candidates": args.output_dir / "figure2_detector_candidates_D1-D4.png",
        "annotated synthetic": args.output_dir / "figure2_synthetic_rbf_D1-D4.png",
    }
    create_ct_stack(volume, outputs["ct stack"])
    create_detector_grid(volume, detections, outputs["detector candidates"])
    create_annotated_synthetic(synthetic_png, detections, outputs["annotated synthetic"])

    for label, path in outputs.items():
        with Image.open(path) as image:
            print(f"{label}: {path} ({image.width}x{image.height}, {image.mode})")


if __name__ == "__main__":
    main()
