#!/usr/bin/env python3
"""
Preprocess LUNA16 MetaImage (.mhd/.raw) CT scans.

Input expected structure, for example:
  LUNA16/
    subsets/subset0/subset0/*.mhd
    subsets/subset1/subset1/*.mhd
    ...
    annotations/annotations.csv

Outputs one folder per scan:
  output_dir/<seriesuid>/
    <seriesuid>_volume.nii.gz
    <seriesuid>_nodule_mask.nii.gz        optional
    <seriesuid>_metadata.json

Also writes:
  output_dir/luna16_labels.csv

The labels CSV format is:
  seriesuid,x,y,z,w,h,d,label

Coordinates x,y,z are voxel centers in the processed output volume space.
Sizes w,h,d are physical sizes in mm.
"""

import argparse
import csv
import json
import math
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import label as connected_components

from lungmask import LMInferer

GLOBAL_INFERER = None


def lumTrans(img: np.ndarray) -> np.ndarray:
    """Clip to lung window [-1200, 600] and convert to uint8 [0, 255]."""
    lungwin = np.array([-1200.0, 600.0], dtype=np.float32)
    newimg = (img.astype(np.float32) - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    return (newimg * 255).astype("uint8")


def process_mask(mask: np.ndarray, iterations: int = 10) -> np.ndarray:
    struct = generate_binary_structure(3, 1)
    return binary_dilation(mask > 0, structure=struct, iterations=iterations)


def get_lung_mask(volume: np.ndarray, model_name: str = "R231", force_cpu: bool = False) -> np.ndarray:
    """Infer lung mask with lungmask. Input/output array order: z,y,x."""

    global GLOBAL_INFERER
    if GLOBAL_INFERER is None:
        GLOBAL_INFERER = LMInferer(modelname=model_name, force_cpu=force_cpu)
    segmentation = GLOBAL_INFERER.apply(volume)
    return (segmentation > 0).astype(np.uint8)


def get_extendbox(mask: np.ndarray, margin: int = 10):
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None

    zz, yy, xx = coords
    box = np.array(
        [
            [np.min(zz), np.max(zz) + 1],
            [np.min(yy), np.max(yy) + 1],
            [np.min(xx), np.max(xx) + 1],
        ],
        dtype=int,
    )
    shape = np.array(mask.shape, dtype=int)
    min_zyx = np.maximum(box[:, 0] - margin, 0)
    max_zyx = np.minimum(box[:, 1] + margin, shape)
    return np.stack([min_zyx, max_zyx], axis=1).astype(int)


def crop_image_to_bbox(image: sitk.Image, extendbox_zyx: np.ndarray) -> sitk.Image:
    """Crop SimpleITK image using z,y,x bbox [[z0,z1],[y0,y1],[x0,x1]]."""
    index_xyz = [
        int(extendbox_zyx[2, 0]),
        int(extendbox_zyx[1, 0]),
        int(extendbox_zyx[0, 0]),
    ]
    size_xyz = [
        int(extendbox_zyx[2, 1] - extendbox_zyx[2, 0]),
        int(extendbox_zyx[1, 1] - extendbox_zyx[1, 0]),
        int(extendbox_zyx[0, 1] - extendbox_zyx[0, 0]),
    ]
    return sitk.RegionOfInterest(image, size_xyz, index_xyz)


def image_from_array_like(array: np.ndarray, reference_image: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference_image)
    return image


def resample_image(
    image: sitk.Image,
    out_spacing=(1.0, 1.0, 1.0),
    interpolator=sitk.sitkLinear,
    output_pixel_type=None,
) -> sitk.Image:
    out_spacing = tuple(float(s) for s in out_spacing)
    in_spacing = image.GetSpacing()
    in_size = image.GetSize()
    out_size = [
        max(1, int(round(in_size[i] * (in_spacing[i] / out_spacing[i]))))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(interpolator)
    if output_pixel_type is not None:
        resampler.SetOutputPixelType(output_pixel_type)
    return resampler.Execute(image)


def write_nifti(image: sitk.Image, output_path, compression=True):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path), useCompression=compression)


def save_metadata(output_dir, seriesuid, image, original_mhd_path=None, extendbox_zyx=None):
    metadata = {
        "seriesuid": seriesuid,
        "original_mhd_path": str(original_mhd_path) if original_mhd_path is not None else None,
        "spacing_xyz": list(map(float, image.GetSpacing())),
        "origin_xyz": list(map(float, image.GetOrigin())),
        "size_xyz": list(map(int, image.GetSize())),
        "direction": list(map(float, image.GetDirection())),
        "extendbox_zyx": None,
    }
    if extendbox_zyx is not None:
        metadata["extendbox_zyx"] = {
            "z_min": int(extendbox_zyx[0, 0]),
            "z_max": int(extendbox_zyx[0, 1]),
            "y_min": int(extendbox_zyx[1, 0]),
            "y_max": int(extendbox_zyx[1, 1]),
            "x_min": int(extendbox_zyx[2, 0]),
            "x_max": int(extendbox_zyx[2, 1]),
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{seriesuid}_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def load_luna_annotations(csv_path):
    """Return dict: seriesuid -> list of nodules with world coordinates and diameter_mm."""
    csv_path = Path(csv_path)
    annotations = {}
    if not csv_path.exists():
        return annotations

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seriesuid = row["seriesuid"]
            annotations.setdefault(seriesuid, []).append(
                {
                    "coordX": float(row["coordX"]),
                    "coordY": float(row["coordY"]),
                    "coordZ": float(row["coordZ"]),
                    "diameter_mm": float(row["diameter_mm"]),
                }
            )
    return annotations


def world_to_voxel_xyz(world_xyz, image: sitk.Image) -> np.ndarray:
    """Convert physical/world xyz coordinate to continuous voxel xyz index."""
    return np.array(image.TransformPhysicalPointToContinuousIndex(tuple(map(float, world_xyz))), dtype=np.float32)


def nodule_mask_from_annotations(image: sitk.Image, nodules) -> np.ndarray:
    """
    Create binary nodule sphere masks from LUNA16 annotations.

    Uses SimpleITK physical coordinate transforms, so origin/spacing/direction are respected.
    Output array order: z,y,x.
    """
    arr_shape_zyx = sitk.GetArrayFromImage(image).shape
    mask = np.zeros(arr_shape_zyx, dtype=np.uint8)

    if not nodules:
        return mask

    spacing_xyz = np.array(image.GetSpacing(), dtype=np.float32)
    radius_margin = 0.5

    for n in nodules:
        center_world = np.array([n["coordX"], n["coordY"], n["coordZ"]], dtype=np.float32)
        radius_mm = float(n["diameter_mm"]) / 2.0

        center_idx_xyz = world_to_voxel_xyz(center_world, image)
        radius_vox_xyz = np.maximum((radius_mm / spacing_xyz) + radius_margin, 1.0)

        x0 = max(0, int(math.floor(center_idx_xyz[0] - radius_vox_xyz[0])))
        x1 = min(arr_shape_zyx[2], int(math.ceil(center_idx_xyz[0] + radius_vox_xyz[0] + 1)))
        y0 = max(0, int(math.floor(center_idx_xyz[1] - radius_vox_xyz[1])))
        y1 = min(arr_shape_zyx[1], int(math.ceil(center_idx_xyz[1] + radius_vox_xyz[1] + 1)))
        z0 = max(0, int(math.floor(center_idx_xyz[2] - radius_vox_xyz[2])))
        z1 = min(arr_shape_zyx[0], int(math.ceil(center_idx_xyz[2] + radius_vox_xyz[2] + 1)))

        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue

        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        # Convert voxel coordinates back to approximate physical distance using spacing.
        dx = (xx - center_idx_xyz[0]) * spacing_xyz[0]
        dy = (yy - center_idx_xyz[1]) * spacing_xyz[1]
        dz = (zz - center_idx_xyz[2]) * spacing_xyz[2]
        sphere = (dx * dx + dy * dy + dz * dz) <= (radius_mm * radius_mm)

        mask[z0:z1, y0:y1, x0:x1] = np.logical_or(mask[z0:z1, y0:y1, x0:x1], sphere)

    return mask.astype(np.uint8)


def labels_from_annotations(seriesuid, image: sitk.Image, nodules):
    """
    Create labels CSV rows from annotations in the processed image coordinate space.
    Row format: seriesuid,x,y,z,w,h,d,label
    """
    rows = []
    spacing_xyz = np.array(image.GetSpacing(), dtype=np.float32)

    for n in nodules:
        center_world = [n["coordX"], n["coordY"], n["coordZ"]]
        center_xyz = world_to_voxel_xyz(center_world, image)
        diameter_mm = float(n["diameter_mm"])
        size_xyz_mm = np.array([diameter_mm, diameter_mm, diameter_mm], dtype=np.float32)

        size_xyz_vox = size_xyz_mm / spacing_xyz
        size_xyz_mm_recomputed = size_xyz_vox * spacing_xyz

        rows.append(
            [
                seriesuid,
                float(center_xyz[0]),
                float(center_xyz[1]),
                float(center_xyz[2]),
                float(size_xyz_mm_recomputed[0]),
                float(size_xyz_mm_recomputed[1]),
                float(size_xyz_mm_recomputed[2]),
                "nodule",
            ]
        )
    return rows


def labels_from_mask_image(seriesuid, mask_image, min_voxels=1):
    mask = sitk.GetArrayFromImage(mask_image)
    spacing_zyx = np.array(mask_image.GetSpacing()[::-1], dtype="float32")
    labeled, num = connected_components(mask > 0)
    rows = []

    for component_id in range(1, num + 1):
        coords = np.argwhere(labeled == component_id)
        if coords.shape[0] < min_voxels:
            continue

        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        center_zyx = np.array(
            [
                (z_min + z_max) / 2.0,
                (y_min + y_max) / 2.0,
                (x_min + x_max) / 2.0,
            ],
            dtype=np.float32,
        )
        size_zyx_vox = np.array(
            [
                z_max - z_min + 1,
                y_max - y_min + 1,
                x_max - x_min + 1,
            ],
            dtype=np.float32,
        )
        size_xyz_mm = (size_zyx_vox * spacing_zyx)[::-1]
        center_xyz = center_zyx[::-1]

        rows.append(
            [
                seriesuid,
                float(center_xyz[0]),
                float(center_xyz[1]),
                float(center_xyz[2]),
                float(size_xyz_mm[0]),
                float(size_xyz_mm[1]),
                float(size_xyz_mm[2]),
                "nodule",
            ]
        )
    return rows


def write_boxes_csv(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as f:
        f.write("seriesuid,x,y,z,w,h,d,label\n")
        for row in rows:
            f.write(
                "{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{}\n".format(
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
                )
            )


def find_luna_scans(luna_root):
    luna_root = Path(luna_root)

    candidates = []
    for base in [luna_root / "subsets", luna_root]:
        if base.exists():
            candidates.extend(base.rglob("*.mhd"))

    scans = []
    seen = set()
    for p in candidates:
        p = p.resolve()
        s = str(p)
        if s in seen:
            continue
        seen.add(s)

        # Exclude masks if the masks folder was extracted into the same root.
        lower = s.lower()
        if "seg-lungs" in lower or "/masks/" in lower or "\\masks\\" in lower:
            continue

        # Keep only scan files that have matching .raw data.
        raw_path = p.with_suffix(".raw")
        if raw_path.exists():
            scans.append(p)

    return sorted(scans)


def process_luna_scan(
    mhd_path,
    output_root,
    annotations_by_series,
    write_mask=True,
    overwrite=False,
    crop_lungs=True,
    lung_padding=10,
    lungmask_model="R231",
    force_cpu=False,
    out_spacing=(1.0, 1.0, 1.0),
):
    mhd_path = Path(mhd_path)
    seriesuid = mhd_path.stem
    output_dir = Path(output_root) / seriesuid
    volume_path = output_dir / f"{seriesuid}_volume.nii.gz"
    mask_path = output_dir / f"{seriesuid}_nodule_mask.nii.gz"

    if not overwrite and volume_path.exists() and (mask_path.exists() or not write_mask):
        print(f"{seriesuid}: already processed")
        label_rows = []
        if write_mask and mask_path.exists():
            label_rows = labels_from_mask_image(seriesuid, sitk.ReadImage(str(mask_path)))
        return str(volume_path), str(mask_path) if mask_path.exists() else None, label_rows

    try:
        image = sitk.ReadImage(str(mhd_path))
    except Exception as e:
        print(f"{seriesuid}: failed to read {mhd_path}: {e}")
        return None, None, []

    image = resample_image(image, out_spacing=out_spacing, interpolator=sitk.sitkLinear)
    img_array = lumTrans(sitk.GetArrayFromImage(image))
    extendbox = None

    if crop_lungs:
        try:
            # Lungmask should run on intensity image, not uint8 lung-windowed image.
            original_resampled_array = sitk.GetArrayFromImage(image)
            lung_mask = get_lung_mask(original_resampled_array, model_name=lungmask_model, force_cpu=force_cpu)
            dilated_mask = process_mask(lung_mask, iterations=10)
            extendbox = get_extendbox(lung_mask, margin=lung_padding)

            if extendbox is not None:
                img_array = img_array * dilated_mask.astype(np.uint8)
                crop_arr = img_array[
                    extendbox[0, 0]:extendbox[0, 1],
                    extendbox[1, 0]:extendbox[1, 1],
                    extendbox[2, 0]:extendbox[2, 1],
                ]
                image_roi = crop_image_to_bbox(image, extendbox)
                image = image_from_array_like(crop_arr, image_roi)
            else:
                print(f"{seriesuid}: no lung bbox found; writing uncropped volume")
                image = image_from_array_like(img_array, image)
        except Exception as e:
            print(f"{seriesuid}: lung crop failed ({e}); writing uncropped volume")
            image = image_from_array_like(img_array, image)
            extendbox = None
    else:
        image = image_from_array_like(img_array, image)

    nodules = annotations_by_series.get(seriesuid, [])
    written_mask = None
    label_rows = labels_from_annotations(seriesuid, image, nodules)

    if write_mask:
        mask_array = nodule_mask_from_annotations(image, nodules)
        mask_image = image_from_array_like(mask_array, image)
        write_nifti(mask_image, mask_path)
        written_mask = str(mask_path)

    write_nifti(image, volume_path)
    save_metadata(output_dir, seriesuid, image, original_mhd_path=mhd_path, extendbox_zyx=extendbox)

    print(
        f"{seriesuid}: wrote {volume_path}"
        + (f" and {mask_path}" if written_mask else "")
        + f" | nodules: {len(nodules)}"
    )
    return str(volume_path), written_mask, label_rows


def preprocess_luna16(
    luna_root,
    output_dir,
    annotations_csv=None,
    num_workers=1,
    limit=None,
    write_mask=True,
    overwrite=False,
    crop_lungs=True,
    lung_padding=10,
    lungmask_model="R231",
    force_cpu=False,
    out_spacing=(1.0, 1.0, 1.0),
    labels_csv=None,
):
    luna_root = Path(luna_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scans = find_luna_scans(luna_root)
    if limit is not None:
        scans = scans[:limit]

    if annotations_csv is None:
        default_paths = [
            luna_root / "annotations" / "annotations.csv",
            luna_root / "annotations.csv",
        ]
        annotations_csv = next((p for p in default_paths if p.exists()), None)

    annotations_by_series = load_luna_annotations(annotations_csv) if annotations_csv else {}

    print(f"starting LUNA16 preprocessing: {len(scans)} scan(s)")
    print(f"loaded annotations for {len(annotations_by_series)} scan(s)")

    kwargs = {
        "output_root": str(output_dir),
        "annotations_by_series": annotations_by_series,
        "write_mask": write_mask,
        "overwrite": overwrite,
        "crop_lungs": crop_lungs,
        "lung_padding": lung_padding,
        "lungmask_model": lungmask_model,
        "force_cpu": force_cpu,
        "out_spacing": out_spacing,
    }

    if num_workers > 1:
        worker = partial(process_luna_scan, **kwargs)
        with Pool(num_workers) as pool:
            results = pool.map(worker, [str(scan) for scan in scans])
    else:
        results = [process_luna_scan(str(scan), **kwargs) for scan in scans]

    label_rows = []
    for _, _, rows in results:
        label_rows.extend(rows)

    labels_csv = Path(labels_csv) if labels_csv is not None else output_dir / "luna16_labels.csv"
    write_boxes_csv(labels_csv, label_rows)
    print(f"wrote labels CSV: {labels_csv}")
    print("end LUNA16 preprocessing")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LUNA16 .mhd/.raw CT scans to NIfTI volumes and annotation-derived nodule masks."
    )
    parser.add_argument("--luna-root", "--dicom-dir", "--lidc-data", required=True, help="Root containing LUNA16 subsets/.mhd files.")
    parser.add_argument("--annotations-csv", default=None, help="Path to LUNA16 annotations.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory where per-scan NIfTI folders will be written.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N sorted scans.")
    parser.add_argument("--volume-only", action="store_true", help="Write only CT volumes, not nodule masks.")
    parser.add_argument("--no-lung-crop", action="store_true", help="Disable lungmask-based lung cropping.")
    parser.add_argument("--lung-padding", type=int, default=10, help="Voxel padding around lung bbox.")
    parser.add_argument("--lungmask-model", default="R231", help="LMInferer model name.")
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0), help="Output spacing in x y z mm.")
    parser.add_argument("--labels-csv", default=None, help="Single output CSV path for all labels.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage for lungmask.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NIfTI files.")
    args = parser.parse_args()

    preprocess_luna16(
        luna_root=args.luna_root,
        output_dir=args.output_dir,
        annotations_csv=args.annotations_csv,
        num_workers=args.num_workers,
        limit=args.limit,
        write_mask=not args.volume_only,
        overwrite=args.overwrite,
        crop_lungs=not args.no_lung_crop,
        lung_padding=args.lung_padding,
        lungmask_model=args.lungmask_model,
        force_cpu=args.cpu,
        out_spacing=args.spacing,
        labels_csv=args.labels_csv,
    )


if __name__ == "__main__":
    main()
