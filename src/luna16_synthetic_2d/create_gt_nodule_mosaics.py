"""Create contact-sheet mosaics of original LUNA16 GT nodule slices."""

from __future__ import annotations

import argparse
import math
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


def resolve_path(processed_dir: Path, raw: str, seriesuid: str, suffix: str) -> Path:
    candidates: list[Path] = []
    if raw and raw != "nan":
        path = Path(raw)
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(processed_dir / path)
            if path.parts and path.parts[0].startswith("subset"):
                candidates.append(processed_dir / Path(*path.parts[1:]))
    candidates.append(processed_dir / seriesuid / f"{seriesuid}_{suffix}.nii.gz")
    for subset_idx in range(10):
        candidates.append(processed_dir / f"subset{subset_idx}" / seriesuid / f"{seriesuid}_{suffix}.nii.gz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve {suffix} for {seriesuid}. Tried {candidates[:4]} ...")


def window_to_uint8(slice_2d: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(slice_2d, low, high)
    return ((clipped - low) / (high - low) * 255.0).astype(np.uint8)


def crop_with_padding(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int, fill: float = 0) -> np.ndarray:
    h, w = arr.shape[:2]
    out_shape = (y1 - y0, x1 - x0, *arr.shape[2:])
    out = np.full(out_shape, fill, dtype=arr.dtype)
    src_y0, src_y1 = max(y0, 0), min(y1, h)
    src_x0, src_x1 = max(x0, 0), min(x1, w)
    dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0
    out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def component_tile(
    volume: np.ndarray,
    component: np.ndarray,
    seriesuid: str,
    component_idx: int,
    tile_size: int,
    crop_size: int,
    margin: int,
    window_center: float,
    window_width: float,
) -> tuple[Image.Image, dict[str, object]]:
    zyx = np.argwhere(component)
    z_values, counts = np.unique(zyx[:, 0], return_counts=True)
    z = int(z_values[np.argmax(counts)])
    slice_mask = component[z]
    yy, xx = np.where(slice_mask)
    y_min, y_max = int(yy.min()), int(yy.max())
    x_min, x_max = int(xx.min()), int(xx.max())
    cy = int(round((y_min + y_max) / 2))
    cx = int(round((x_min + x_max) / 2))

    half = max(crop_size // 2, (max(y_max - y_min + 1, x_max - x_min + 1) // 2) + margin)
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    crop = crop_with_padding(window_to_uint8(volume[z], window_center, window_width), y0, y1, x0, x1, fill=0)
    mask_crop = crop_with_padding(slice_mask.astype(np.uint8), y0, y1, x0, x1, fill=0).astype(bool)
    base = Image.fromarray(crop, mode="L").convert("RGB").resize((tile_size, tile_size), Image.Resampling.BILINEAR)

    overlay_mask = Image.fromarray((mask_crop.astype(np.uint8) * 255), mode="L").resize(
        (tile_size, tile_size), Image.Resampling.NEAREST
    )
    edge = np.logical_xor(mask_crop, ndi.binary_erosion(mask_crop))
    edge_img = Image.fromarray((edge.astype(np.uint8) * 255), mode="L").resize((tile_size, tile_size), Image.Resampling.NEAREST)

    red = Image.new("RGB", (tile_size, tile_size), (255, 0, 0))
    base = Image.composite(Image.blend(base, red, 0.25), base, overlay_mask)
    draw = ImageDraw.Draw(base)
    edge_arr = np.asarray(edge_img) > 0
    if edge_arr.any():
        ys, xs = np.where(edge_arr)
        for px, py in zip(xs.tolist(), ys.tolist()):
            draw.point((px, py), fill=(255, 230, 0))

    scale = tile_size / float(2 * half)
    box = [
        int(round((x_min - x0) * scale)),
        int(round((y_min - y0) * scale)),
        int(round((x_max + 1 - x0) * scale)),
        int(round((y_max + 1 - y0) * scale)),
    ]
    draw.rectangle(box, outline=(0, 255, 255), width=2)
    title = f"{component_idx}: z={z} vox={len(zyx)}"
    draw.rectangle((0, 0, tile_size, 18), fill=(0, 0, 0))
    draw.text((4, 3), title, fill=(255, 255, 255), font=ImageFont.load_default())

    row = {
        "seriesuid": seriesuid,
        "component_idx": component_idx,
        "z": z,
        "center_y": cy,
        "center_x": cx,
        "bbox_y_min": y_min,
        "bbox_y_max": y_max,
        "bbox_x_min": x_min,
        "bbox_x_max": x_max,
        "voxel_count": int(len(zyx)),
    }
    return base, row


def load_unique_split_rows(csv_files: list[Path], split: str | None) -> pd.DataFrame:
    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["source_csv"] = str(csv_file)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    if split and "split" in data.columns:
        data = data[data["split"].astype(str) == split].copy()
    if "nodule_count" in data.columns:
        data = data[data["nodule_count"].fillna(0).astype(int) > 0].copy()
    return data.drop_duplicates("seriesuid").reset_index(drop=True)


def save_mosaics(tiles: list[Image.Image], rows: list[dict[str, object]], out_dir: Path, cols: int, rows_per_page: int) -> list[Path]:
    page_paths: list[Path] = []
    if not tiles:
        return page_paths
    tile_w, tile_h = tiles[0].size
    page_size = cols * rows_per_page
    for page_idx in range(math.ceil(len(tiles) / page_size)):
        page_tiles = tiles[page_idx * page_size : (page_idx + 1) * page_size]
        mosaic = Image.new("RGB", (cols * tile_w, rows_per_page * tile_h), (20, 20, 20))
        for idx, tile in enumerate(page_tiles):
            x = (idx % cols) * tile_w
            y = (idx // cols) * tile_h
            mosaic.paste(tile, (x, y))
        path = out_dir / f"gt_nodule_mosaic_page_{page_idx + 1:03d}.jpg"
        mosaic.save(path, quality=92)
        page_paths.append(path)
    pd.DataFrame(rows).to_csv(out_dir / "gt_nodule_mosaic_index.csv", index=False)
    return page_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/LUNA16_preprocessed"))
    parser.add_argument("--csv-glob", default="data/LUNA16_preprocessed/cv_splits/luna16_classification_fold*.csv")
    parser.add_argument("--split", default="test", help="Split to visualize; use empty string for all rows.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/luna16_gt_nodule_mosaics"))
    parser.add_argument("--max-scans", type=int, default=0, help="Limit scans for quick previews; 0 means all.")
    parser.add_argument("--tile-size", type=int, default=192)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--margin", type=int, default=24)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--window-center", type=float, default=-600.0)
    parser.add_argument("--window-width", type=float, default=1500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(Path(".").glob(args.csv_glob))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched {args.csv_glob}")

    split = args.split if args.split else None
    rows_df = load_unique_split_rows(csv_files, split)
    if args.max_scans > 0:
        rows_df = rows_df.head(args.max_scans)

    tiles: list[Image.Image] = []
    rows: list[dict[str, object]] = []
    for _, row in rows_df.iterrows():
        seriesuid = str(row["seriesuid"])
        volume_path = resolve_path(args.processed_dir, str(row.get("image_path", "")), seriesuid, "volume")
        mask_path = resolve_path(args.processed_dir, str(row.get("nodule_mask_path", "")), seriesuid, "nodule_mask")
        volume = load_zyx(volume_path)
        mask = load_zyx(mask_path) > 0
        labeled, n_components = ndi.label(mask)
        for component_idx in range(1, n_components + 1):
            component = labeled == component_idx
            if not component.any():
                continue
            tile, out_row = component_tile(
                volume=volume,
                component=component,
                seriesuid=seriesuid,
                component_idx=component_idx,
                tile_size=args.tile_size,
                crop_size=args.crop_size,
                margin=args.margin,
                window_center=args.window_center,
                window_width=args.window_width,
            )
            out_row.update(
                {
                    "volume_path": str(volume_path),
                    "mask_path": str(mask_path),
                    "split": str(row.get("split", "")),
                    "target": row.get("target", ""),
                    "target_name": row.get("target_name", ""),
                    "nodule_count_csv": row.get("nodule_count", ""),
                    "max_nodule_mean_malignancy": row.get("max_nodule_mean_malignancy", ""),
                }
            )
            tiles.append(tile)
            rows.append(out_row)

    page_paths = save_mosaics(tiles, rows, args.out_dir, args.cols, args.rows)
    summary = pd.DataFrame(
        [
            {
                "split": split or "all",
                "scans": int(len(rows_df)),
                "nodule_components": int(len(rows)),
                "pages": int(len(page_paths)),
                "out_dir": str(args.out_dir),
            }
        ]
    )
    summary.to_csv(args.out_dir / "gt_nodule_mosaic_summary.csv", index=False)
    print(
        f"Wrote {len(page_paths)} mosaic pages with {len(rows)} GT nodule components "
        f"from {len(rows_df)} scans to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
