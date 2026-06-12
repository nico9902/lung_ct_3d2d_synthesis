import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage


LABEL_COLUMNS = ["seriesuid", "image_path", "split", "x", "y", "z", "w", "h", "d", "label"]


def patient_dirs(processed_dir):
    return sorted(
        path
        for path in Path(processed_dir).iterdir()
        if path.is_dir() and path.name.startswith("LIDC-IDRI")
    )


def find_image_path(patient_dir, patient_id):
    candidates = [
        patient_dir / f"{patient_id}_volume.nii.gz",
        patient_dir / "volume.npy",
        patient_dir / f"{patient_id}_volume.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return ""


def load_split_map(split_csv):
    if split_csv is None:
        return {}

    df = pd.read_csv(split_csv)
    id_col = next((col for col in ["seriesuid", "patient_id", "id"] if col in df.columns), None)
    if id_col is None or "split" not in df.columns:
        raise ValueError("Split CSV must contain a patient id column and a 'split' column.")

    return dict(zip(df[id_col].astype(str), df["split"].astype(str)))


def bbox_to_row(patient_id, image_path, split, bbox):
    x_min, x_max = int(bbox["x_min"]), int(bbox["x_max"])
    y_min, y_max = int(bbox["y_min"]), int(bbox["y_max"])
    z_min, z_max = int(bbox["z_min"]), int(bbox["z_max"])

    w = x_max - x_min
    h = y_max - y_min
    d = z_max - z_min
    if w <= 0 or h <= 0 or d <= 0:
        return None

    return {
        "seriesuid": patient_id,
        "image_path": str(image_path),
        "split": split,
        "x": x_min + (w - 1) / 2.0,
        "y": y_min + (h - 1) / 2.0,
        "z": z_min + (d - 1) / 2.0,
        "w": float(w),
        "h": float(h),
        "d": float(d),
        "label": "nodule",
    }


def rows_from_nodule_info(patient_dir, patient_id, image_path, split):
    info_path = patient_dir / "nodule_mask" / "nodule_info.json"
    if not info_path.exists():
        return None

    with info_path.open() as f:
        nodules = json.load(f)

    rows = []
    for nodule in nodules:
        bbox = nodule.get("processed_bbox") or nodule.get("original_bbox")
        if not bbox:
            continue
        row = bbox_to_row(patient_id, image_path, split, bbox)
        if row is not None:
            rows.append(row)

    return rows


def load_mask(patient_dir, patient_id):
    npy_candidates = [
        patient_dir / "nodule_mask.npy",
        patient_dir / f"{patient_id}_nodule_mask.npy",
        patient_dir / "nodule_mask" / "mask_volume.npy",
    ]
    for path in npy_candidates:
        if path.exists():
            return np.load(path)

    nii_candidates = [
        patient_dir / f"{patient_id}_nodule_mask.nii.gz",
        patient_dir / "nodule_mask.nii.gz",
    ]
    for path in nii_candidates:
        if path.exists():
            return load_nii_mask(path)

    mask_dir = patient_dir / "nodule_mask"
    slice_paths = sorted(mask_dir.glob("nodule_mask_*.npy"))
    if slice_paths:
        return np.stack([np.load(path) for path in slice_paths], axis=0)

    return None


def load_nii_mask(path):
    try:
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    except ImportError:
        pass

    try:
        import nibabel as nib

        data = nib.load(str(path)).get_fdata()
        return np.asarray(data).transpose(2, 1, 0)
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot read {path}. Install SimpleITK or nibabel, or use processed npy masks."
        ) from exc


def rows_from_mask(patient_dir, patient_id, image_path, split):
    mask = load_mask(patient_dir, patient_id)
    if mask is None:
        return []

    mask = np.asarray(mask) > 0
    if mask.ndim != 3 or not mask.any():
        return []

    labeled, num_features = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 3))
    objects = ndimage.find_objects(labeled)

    rows = []
    for obj in objects[:num_features]:
        if obj is None:
            continue
        z_slice, y_slice, x_slice = obj
        bbox = {
            "z_min": z_slice.start,
            "z_max": z_slice.stop,
            "y_min": y_slice.start,
            "y_max": y_slice.stop,
            "x_min": x_slice.start,
            "x_max": x_slice.stop,
        }
        row = bbox_to_row(patient_id, image_path, split, bbox)
        if row is not None:
            rows.append(row)

    return rows


def build_labels(processed_dir, split_csv=None, default_split="train", prefer_nodule_info=True):
    split_map = load_split_map(split_csv)
    rows = []

    for patient_dir in patient_dirs(processed_dir):
        patient_id = patient_dir.name
        split = split_map.get(patient_id, default_split)
        image_path = find_image_path(patient_dir, patient_id)

        patient_rows = None
        if prefer_nodule_info:
            patient_rows = rows_from_nodule_info(patient_dir, patient_id, image_path, split)

        if patient_rows is None:
            patient_rows = rows_from_mask(patient_dir, patient_id, image_path, split)

        rows.extend(patient_rows)

    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def main():
    parser = argparse.ArgumentParser(
        description="Generate lidc_labels.csv from processed LIDC patient folders."
    )
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output", default=None)
    parser.add_argument("--split_csv", default=None)
    parser.add_argument("--default_split", default="train")
    parser.add_argument(
        "--from-mask",
        action="store_true",
        help="Ignore nodule_info.json and derive boxes from connected components in masks.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output = Path(args.output) if args.output else processed_dir / "lidc_labels.csv"

    labels = build_labels(
        processed_dir=processed_dir,
        split_csv=args.split_csv,
        default_split=args.default_split,
        prefer_nodule_info=not args.from_mask,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output, index=False)

    print(f"Saved {len(labels)} annotations to {output}")
    if len(labels):
        print(f"Patients with annotations: {labels['seriesuid'].nunique()}")


if __name__ == "__main__":
    main()
