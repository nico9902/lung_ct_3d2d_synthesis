"""Export per-sample GT/RBF/Shepard nodule crops with GT mask overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi


def load_zyx(path: Path) -> np.ndarray:
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume at {path}, got shape {arr.shape}.")
    return arr.transpose(2, 1, 0)


def window_to_uint8(slice_2d: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(slice_2d, low, high)
    return ((clipped - low) / (high - low) * 255.0).astype(np.uint8)


def crop_with_padding(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int, fill: int = 0) -> np.ndarray:
    h, w = arr.shape[:2]
    out_shape = (y1 - y0, x1 - x0, *arr.shape[2:])
    out = np.full(out_shape, fill, dtype=arr.dtype)
    src_y0, src_y1 = max(y0, 0), min(y1, h)
    src_x0, src_x1 = max(x0, 0), min(x1, w)
    dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0
    out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def synthetic_surface_path(root: Path, seriesuid: str) -> Path:
    return root / seriesuid / f"surface_{seriesuid}.png"


def overlay_panel(
    arr: np.ndarray,
    mask: np.ndarray,
    label: str,
    panel_size: int,
    box: tuple[int, int, int, int],
) -> Image.Image:
    base = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB").resize(
        (panel_size, panel_size), Image.Resampling.BILINEAR
    )
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
        (panel_size, panel_size), Image.Resampling.NEAREST
    )
    edge = np.logical_xor(mask, ndi.binary_erosion(mask))
    edge_img = Image.fromarray((edge.astype(np.uint8) * 255), mode="L").resize(
        (panel_size, panel_size), Image.Resampling.NEAREST
    )
    red = Image.new("RGB", (panel_size, panel_size), (255, 0, 0))
    base = Image.composite(Image.blend(base, red, 0.28), base, mask_img)

    draw = ImageDraw.Draw(base)
    edge_arr = np.asarray(edge_img) > 0
    if edge_arr.any():
        ys, xs = np.where(edge_arr)
        for px, py in zip(xs.tolist(), ys.tolist()):
            draw.point((px, py), fill=(255, 230, 0))
    draw.rectangle(box, outline=(0, 255, 255), width=2)
    draw.rectangle((0, 0, panel_size, 18), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return base


def build_sample_images(
    row: pd.Series,
    volume: np.ndarray,
    labeled_mask: np.ndarray,
    rbf_surface: np.ndarray,
    shepard_surface: np.ndarray,
    panel_size: int,
    crop_size: int,
    margin: int,
    window_center: float,
    window_width: float,
) -> dict[str, Image.Image]:
    z = int(row["z"])
    component_idx = int(row["component_idx"])
    y_min = int(row["bbox_y_min"])
    y_max = int(row["bbox_y_max"])
    x_min = int(row["bbox_x_min"])
    x_max = int(row["bbox_x_max"])
    cy = int(row["center_y"])
    cx = int(row["center_x"])

    half = max(crop_size // 2, (max(y_max - y_min + 1, x_max - x_min + 1) // 2) + margin)
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    scale = panel_size / float(2 * half)
    box = (
        int(round((x_min - x0) * scale)),
        int(round((y_min - y0) * scale)),
        int(round((x_max + 1 - x0) * scale)),
        int(round((y_max + 1 - y0) * scale)),
    )

    component = labeled_mask == component_idx
    mask_crop = crop_with_padding(component[z].astype(np.uint8), y0, y1, x0, x1, fill=0).astype(bool)
    gt_crop = crop_with_padding(window_to_uint8(volume[z], window_center, window_width), y0, y1, x0, x1, fill=0)
    rbf_crop = crop_with_padding(rbf_surface, y0, y1, x0, x1, fill=0)
    shepard_crop = crop_with_padding(shepard_surface, y0, y1, x0, x1, fill=0)

    gt = overlay_panel(gt_crop, mask_crop, "GT CT", panel_size, box)
    rbf = overlay_panel(rbf_crop, mask_crop, "RBF synth", panel_size, box)
    shepard = overlay_panel(shepard_crop, mask_crop, "Shepard synth", panel_size, box)

    header_h = 28
    comparison = Image.new("RGB", (panel_size * 3, panel_size + header_h), (16, 16, 16))
    draw = ImageDraw.Draw(comparison)
    title = f"{row['seriesuid']} | comp={component_idx} z={z} vox={int(row['voxel_count'])}"
    draw.text((4, 8), title[:110], fill=(255, 255, 255), font=ImageFont.load_default())
    comparison.paste(gt, (0, header_h))
    comparison.paste(rbf, (panel_size, header_h))
    comparison.paste(shepard, (panel_size * 2, header_h))

    return {
        "gt_ct_overlay": gt,
        "rbf_overlay": rbf,
        "shepard_overlay": shepard,
        "comparison": comparison,
    }


def safe_sample_name(seriesuid: str, component_idx: int) -> str:
    return f"{seriesuid}__component_{component_idx:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-csv", type=Path, default=Path("docs/luna16_gt_nodule_mosaics/gt_nodule_mosaic_index.csv"))
    parser.add_argument("--rbf-root", type=Path, default=Path("data/luna16_saliency_synthetic_detector_top7_minprob0.3_rbf"))
    parser.add_argument("--shepard-root", type=Path, default=Path("data/luna16_saliency_synthetic_detector_top7_minprob0.3_shepard"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/luna16_gt_vs_top7_synthetic_by_sample"))
    parser.add_argument("--max-nodules", type=int, default=0, help="Limit nodule components; 0 means all.")
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--margin", type=int, default=24)
    parser.add_argument("--window-center", type=float, default=-600.0)
    parser.add_argument("--window-width", type=float, default=1500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index_csv)
    if args.max_nodules > 0:
        index = index.head(args.max_nodules).copy()

    current_series = None
    volume = labeled_mask = rbf_surface = shepard_surface = None
    exported_rows: list[dict[str, object]] = []

    for _, row in index.iterrows():
        seriesuid = str(row["seriesuid"])
        component_idx = int(row["component_idx"])
        rbf_path = synthetic_surface_path(args.rbf_root, seriesuid)
        shepard_path = synthetic_surface_path(args.shepard_root, seriesuid)
        if not rbf_path.exists() or not shepard_path.exists():
            continue

        if current_series != seriesuid:
            volume = load_zyx(Path(row["volume_path"]))
            mask = load_zyx(Path(row["mask_path"])) > 0
            labeled_mask, _ = ndi.label(mask)
            rbf_surface = np.asarray(Image.open(rbf_path).convert("L"), dtype=np.uint8)
            shepard_surface = np.asarray(Image.open(shepard_path).convert("L"), dtype=np.uint8)
            current_series = seriesuid

        assert volume is not None and labeled_mask is not None and rbf_surface is not None and shepard_surface is not None
        images = build_sample_images(
            row=row,
            volume=volume,
            labeled_mask=labeled_mask,
            rbf_surface=rbf_surface,
            shepard_surface=shepard_surface,
            panel_size=args.panel_size,
            crop_size=args.crop_size,
            margin=args.margin,
            window_center=args.window_center,
            window_width=args.window_width,
        )

        sample_name = safe_sample_name(seriesuid, component_idx)
        sample_dir = args.out_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for stem, image in images.items():
            path = sample_dir / f"{stem}.jpg"
            image.save(path, quality=94)
            paths[stem] = str(path)

        metadata = {
            "seriesuid": seriesuid,
            "component_idx": component_idx,
            "z": int(row["z"]),
            "center_y": int(row["center_y"]),
            "center_x": int(row["center_x"]),
            "bbox_y_min": int(row["bbox_y_min"]),
            "bbox_y_max": int(row["bbox_y_max"]),
            "bbox_x_min": int(row["bbox_x_min"]),
            "bbox_x_max": int(row["bbox_x_max"]),
            "voxel_count": int(row["voxel_count"]),
            "target": row.get("target", ""),
            "target_name": row.get("target_name", ""),
            "nodule_count_csv": row.get("nodule_count_csv", ""),
            "max_nodule_mean_malignancy": row.get("max_nodule_mean_malignancy", ""),
            "volume_path": row["volume_path"],
            "mask_path": row["mask_path"],
            "rbf_surface": str(rbf_path),
            "shepard_surface": str(shepard_path),
            **paths,
        }
        pd.Series(metadata).to_json(sample_dir / "metadata.json", indent=2)
        exported_rows.append({"sample_dir": str(sample_dir), **metadata})

    pd.DataFrame(exported_rows).to_csv(args.out_dir / "sample_index.csv", index=False)
    pd.DataFrame(
        [
            {
                "samples": len(exported_rows),
                "out_dir": str(args.out_dir),
                "rbf_root": str(args.rbf_root),
                "shepard_root": str(args.shepard_root),
            }
        ]
    ).to_csv(args.out_dir / "sample_summary.csv", index=False)
    print(f"Wrote {len(exported_rows)} sample folders to {args.out_dir}")


if __name__ == "__main__":
    main()
