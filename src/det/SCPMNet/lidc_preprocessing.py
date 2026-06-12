import argparse
import configparser
import json
import math
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import label as connected_components

# Compatibility shims for older pylidc releases on newer Python/numpy stacks.
np.int = int
np.float = float
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

def boxes_from_mask_csv(mask: np.ndarray, spacing_zyx: np.ndarray, min_voxels: int = 1):
    labeled, num = connected_components(mask > 0)
    rows = []

    for component_id in range(1, num + 1):
        coords = np.argwhere(labeled == component_id)

        if coords.shape[0] < min_voxels:
            continue

        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        center_zyx = np.array([
            (z_min + z_max) / 2.0,
            (y_min + y_max) / 2.0,
            (x_min + x_max) / 2.0,
        ], dtype="float32")

        size_zyx_vox = np.array([
            z_max - z_min + 1,
            y_max - y_min + 1,
            x_max - x_min + 1,
        ], dtype="float32")

        size_zyx_mm = size_zyx_vox * spacing_zyx

        # Convert from z,y,x / d,h,w to x,y,z / w,h,d
        center_xyz = center_zyx[::-1]
        size_xyz = size_zyx_mm[::-1]

        label = "nodule"

        rows.append([
            float(center_xyz[0]),
            float(center_xyz[1]),
            float(center_xyz[2]),
            float(size_xyz[0]),
            float(size_xyz[1]),
            float(size_xyz[2]),
            label,
        ])

    return rows


def inertia_ellipsoid_rows_from_instance_masks_csv(instance_masks, spacing_zyx: np.ndarray, min_voxels: int = 1):
    rows = []
    voxel_volume_mm3 = float(np.prod(spacing_zyx))

    for instance_mask in instance_masks:
        coords = np.argwhere(instance_mask > 0)
        if coords.shape[0] < min_voxels:
            continue

        center_zyx = coords.mean(axis=0).astype("float32")
        center_xyz = center_zyx[::-1]

        volume_mm3 = float(coords.shape[0]) * voxel_volume_mm3
        equivalent_sphere_diameter_mm = 2.0 * ((3.0 * volume_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0)

        diameter_mm = equivalent_sphere_diameter_mm
        if coords.shape[0] > 1:
            offsets_mm = (coords.astype("float32") - center_zyx) * spacing_zyx
            covariance = (offsets_mm.T @ offsets_mm) / float(coords.shape[0])
            eigenvalues = np.linalg.eigvalsh(covariance)
            axis_diameters_mm = 2.0 * np.sqrt(np.maximum(5.0 * eigenvalues, 0.0))
            if np.all(axis_diameters_mm > 0):
                diameter_mm = float(np.prod(axis_diameters_mm) ** (1.0 / 3.0))

        rows.append([
            float(center_xyz[0]),
            float(center_xyz[1]),
            float(center_xyz[2]),
            float(diameter_mm),
            float(diameter_mm),
            float(diameter_mm),
            float(diameter_mm),
            "nodule",
        ])

    return rows


def write_boxes_csv(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    has_diameter = any(len(row) == 9 for row in rows)
    with csv_path.open("w") as f:
        if has_diameter:
            f.write("seriesuid,x,y,z,w,h,d,diameter_mm,label\n")
        else:
            f.write("seriesuid,x,y,z,w,h,d,label\n")
        for row in rows:
            if len(row) == 9:
                f.write(
                    "{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{}\n".format(
                        row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]
                    )
                )
            else:
                f.write(
                    "{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{}\n".format(
                        row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
                    )
                )

def _is_dicom_file(path):
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
    except RuntimeError:
        return False
    return reader.HasMetaDataKey("0008|0060") and reader.GetMetaData("0008|0060").strip() == "CT"


def _series_key(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    if not reader.HasMetaDataKey("0020|000e"):
        return None
    return reader.GetMetaData("0020|000e").strip()


def _dicom_sort_key(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    if reader.HasMetaDataKey("0020|0032"):
        position = reader.GetMetaData("0020|0032").split("\\")
        return float(position[2])
    if reader.HasMetaDataKey("0020|0013"):
        return float(reader.GetMetaData("0020|0013"))
    return float("inf")


def find_ct_series(patient_dir):
    """Return the largest CT DICOM series under a possibly nested LIDC patient folder."""
    series = defaultdict(list)
    for root, _, files in os.walk(patient_dir):
        for filename in files:
            path = Path(root) / filename
            if not path.is_file() or not _is_dicom_file(path):
                continue
            key = _series_key(path)
            if key is not None:
                series[key].append(str(path))

    if not series:
        return []
    return max(series.values(), key=len)


def read_dicom_series(dicom_files):
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    sorted_files = reader.GetGDCMSeriesFileNames(os.path.dirname(dicom_files[0]))

    # GetGDCMSeriesFileNames only sees one directory. LIDC trees can be nested, so
    # fall back to the recursively collected files when the selected series spans
    # a different set.
    if set(sorted_files) != set(dicom_files):
        sorted_files = sorted(dicom_files, key=_dicom_sort_key)

    reader.SetFileNames(sorted_files)
    image = reader.Execute()
    z_coords = []
    for i in range(len(sorted_files)):
        if reader.HasMetaDataKey(i, "0020|0032"):
            position = reader.GetMetaData(i, "0020|0032").split("\\")
            z_coords.append(float(position[2]))
        else:
            z_coords.append(float(i))
    return image, z_coords


def _matching_pylidc_scan(patient_id, depth):
    import pylidc as pl

    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
    if not scans:
        return None
    for scan in scans:
        if len(scan.slice_zvals) == depth:
            return scan
        if scan.to_volume().shape[2] == depth:
            return scan
    return scans[0]


def _nearest_z_index(z_to_idx, z_value, tolerance=1e-2):
    rounded = float(np.round(z_value, 4))
    if rounded in z_to_idx:
        return z_to_idx[rounded]

    keys = list(z_to_idx.keys())
    if not keys:
        return None
    key_array = np.array(keys, dtype=np.float64)
    nearest = int(np.argmin(np.abs(key_array - rounded)))
    if abs(keys[nearest] - rounded) <= tolerance:
        return z_to_idx[keys[nearest]]
    return None


def generate_nodule_masks(patient_id, volume_shape, z_coords, consensus_threshold=1):
    """Build union and per-cluster nodule masks in SimpleITK array order: (z, y, x)."""
    import pylidc.utils

    scan = _matching_pylidc_scan(patient_id, volume_shape[0])
    if scan is None:
        print(f"{patient_id}: no pylidc scan found; writing volume only")
        return None

    scan_z_coords = [float(z) for z in scan.slice_zvals]
    z_to_idx = {float(np.round(z, 4)): i for i, z in enumerate(z_coords)}
    mask = np.zeros(volume_shape, dtype=np.uint8)
    instance_masks = []

    for nodule_cluster in scan.cluster_annotations():
        consensus = pylidc.utils.consensus(nodule_cluster, clevel=consensus_threshold)
        if len(consensus) == 2:
            cmask, cbbox = consensus
        else:
            cmask, cbbox, _ = consensus

        instance_mask = np.zeros(volume_shape, dtype=np.uint8)
        y_slice, x_slice, z_slice = cbbox
        for scan_z_idx in range(z_slice.start, z_slice.stop):
            volume_z_idx = _nearest_z_index(z_to_idx, scan_z_coords[scan_z_idx])
            if volume_z_idx is None:
                continue
            local_z_idx = scan_z_idx - z_slice.start
            slice_mask = cmask[:, :, local_z_idx]
            instance_mask[volume_z_idx, y_slice, x_slice] = np.logical_or(
                instance_mask[volume_z_idx, y_slice, x_slice],
                slice_mask,
            )

        if np.any(instance_mask):
            instance_masks.append(instance_mask.astype(np.uint8))
            mask = np.logical_or(mask, instance_mask)

    return mask.astype(np.uint8), instance_masks


def generate_nodule_mask(patient_id, volume_shape, z_coords, consensus_threshold=1):
    """Build a binary nodule mask in SimpleITK array order: (z, y, x)."""
    result = generate_nodule_masks(patient_id, volume_shape, z_coords, consensus_threshold)
    if result is None:
        return None
    mask, _ = result
    return mask


def resample_and_crop_mask_array(mask_array, reference_image, out_spacing, extendbox=None):
    mask_image = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    mask_image.CopyInformation(reference_image)
    mask_image = resample_image(
        mask_image,
        out_spacing=out_spacing,
        interpolator=sitk.sitkNearestNeighbor,
        output_pixel_type=sitk.sitkUInt8,
    )
    if extendbox is not None:
        mask_image = crop_image_to_bbox(mask_image, (extendbox[:, 0], extendbox[:, 1]))
    return mask_image


def resample_mask_array_to_reference(mask_array, source_image, reference_image):
    mask_image = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    mask_image.CopyInformation(source_image)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    return resampler.Execute(mask_image)


def process_mask(mask):
    struct = generate_binary_structure(3, 1)
    return binary_dilation(mask, structure=struct, iterations=10)


def lumTrans(img):
    lungwin = np.array([-1200.0, 600.0])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    return (newimg * 255).astype("uint8")


def get_lung_mask(volume, model_name="R231", force_cpu=False):
    from lungmask import LMInferer

    inferer = LMInferer(modelname=model_name, force_cpu=force_cpu)
    segmentation = inferer.apply(volume)
    return (segmentation > 0).astype(np.uint8)


def get_lung_bbox(lung_mask, padding=10):
    coords = np.argwhere(lung_mask > 0)
    if coords.size == 0:
        return None

    min_zyx = coords.min(axis=0)
    max_zyx = coords.max(axis=0) + 1
    shape_zyx = np.array(lung_mask.shape)
    min_zyx = np.maximum(min_zyx - padding, 0)
    max_zyx = np.minimum(max_zyx + padding, shape_zyx)
    return min_zyx.astype(int), max_zyx.astype(int)


def get_extendbox(mask, margin=10):
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None

    zz, yy, xx = coords
    box = np.array(
        [
            [np.min(zz), np.max(zz)],
            [np.min(yy), np.max(yy)],
            [np.min(xx), np.max(xx)],
        ]
    )
    shape = np.array(mask.shape)
    return np.vstack(
        [
            np.max([[0, 0, 0], box[:, 0] - margin], axis=0),
            np.min([shape, box[:, 1] + margin], axis=0).T,
        ]
    ).T.astype(int)


def crop_image_to_bbox(image, bbox_zyx):
    min_zyx, max_zyx = bbox_zyx
    index_xyz = [int(min_zyx[2]), int(min_zyx[1]), int(min_zyx[0])]
    size_xyz = [
        int(max_zyx[2] - min_zyx[2]),
        int(max_zyx[1] - min_zyx[1]),
        int(max_zyx[0] - min_zyx[0]),
    ]
    return sitk.RegionOfInterest(image, size_xyz, index_xyz)


def resample_image(image, out_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear, output_pixel_type=None):
    out_spacing = tuple(float(spacing) for spacing in out_spacing)
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


def image_from_array_like(array, reference_image):
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference_image)
    return image


def save_metadata(output_dir, patient_id, image, lung_bbox_zyx=None, extendbox_zyx=None):
    metadata = {
        "patient_id": patient_id,
        "spacing_xyz": list(map(float, image.GetSpacing())),
        "origin_xyz": list(map(float, image.GetOrigin())),
        "size_xyz": list(map(int, image.GetSize())),
        "lung_bbox_zyx": None,
        "extendbox_zyx": None,
    }
    if lung_bbox_zyx is not None:
        min_zyx, max_zyx = lung_bbox_zyx
        metadata["lung_bbox_zyx"] = {
            "z_min": int(min_zyx[0]),
            "z_max": int(max_zyx[0]),
            "y_min": int(min_zyx[1]),
            "y_max": int(max_zyx[1]),
            "x_min": int(min_zyx[2]),
            "x_max": int(max_zyx[2]),
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
    with (output_dir / f"{patient_id}_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def write_nifti(image, output_path, compression=True):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix not in {".nii", ".gz"}:
        output_path = output_path.with_suffix(".nii.gz" if compression else ".nii")
    sitk.WriteImage(image, str(output_path), useCompression=compression)


def labels_from_mask_image(patient_id, mask_image, min_voxels=1):
    mask_array = sitk.GetArrayFromImage(mask_image)
    spacing_zyx = np.array(mask_image.GetSpacing()[::-1], dtype="float32")
    rows = boxes_from_mask_csv(mask_array, spacing_zyx=spacing_zyx, min_voxels=min_voxels)
    return [[patient_id] + row for row in rows]


def labels_from_instance_mask_images(patient_id, mask_images, min_voxels=1):
    if not mask_images:
        return []
    spacing_zyx = np.array(mask_images[0].GetSpacing()[::-1], dtype="float32")
    mask_arrays = [sitk.GetArrayFromImage(mask_image) for mask_image in mask_images]
    rows = inertia_ellipsoid_rows_from_instance_masks_csv(mask_arrays, spacing_zyx=spacing_zyx, min_voxels=min_voxels)
    return [[patient_id] + row for row in rows]


def process_lidc_case(
    patient_dir,
    output_root,
    consensus_threshold=1,
    write_mask=True,
    overwrite=False,
    crop_lungs=True,
    lung_padding=10,
    lungmask_model="R231",
    force_cpu=False,
    out_spacing=(1.0, 1.0, 1.0),
    labels_only=False,
):
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name
    output_dir = Path(output_root) / patient_id
    volume_path = output_dir / f"{patient_id}_volume.nii.gz"
    mask_path = output_dir / f"{patient_id}_nodule_mask.nii.gz"

    if not labels_only and not overwrite and volume_path.exists() and (mask_path.exists() or not write_mask):
        print(f"{patient_id}: already processed")
        label_rows = []
        if write_mask and mask_path.exists():
            label_rows = labels_from_mask_image(patient_id, sitk.ReadImage(str(mask_path)))
        return str(volume_path), str(mask_path) if mask_path.exists() else None, label_rows

    if labels_only and not volume_path.exists():
        print(f"{patient_id}: no existing processed volume found at {volume_path}")
        return None, None, []

    dicom_files = find_ct_series(patient_dir)
    if not dicom_files:
        print(f"{patient_id}: no CT DICOM series found")
        return None, None, []

    image, z_coords = read_dicom_series(dicom_files)
    full_image = image
    lung_bbox_zyx = None

    if labels_only:
        saved_volume_image = sitk.ReadImage(str(volume_path))
        label_rows = []
        written_mask = str(mask_path) if mask_path.exists() else None
        if write_mask:
            full_shape = sitk.GetArrayFromImage(full_image).shape
            mask_result = generate_nodule_masks(patient_id, full_shape, z_coords, consensus_threshold)
            if mask_result is not None:
                mask_array, instance_masks = mask_result
                mask_image = resample_mask_array_to_reference(mask_array, full_image, saved_volume_image)
                instance_mask_images = [
                    resample_mask_array_to_reference(instance_mask, full_image, saved_volume_image)
                    for instance_mask in instance_masks
                ]
                label_rows = labels_from_instance_mask_images(patient_id, instance_mask_images)
                write_nifti(mask_image, mask_path)
                written_mask = str(mask_path)
        print(
            f"{patient_id}: reused {volume_path}"
            + (f", wrote {mask_path}" if written_mask else "")
            + f" | labels: {len(label_rows)}"
        )
        return str(volume_path), written_mask, label_rows

    if crop_lungs:
        volume = sitk.GetArrayFromImage(full_image)
        lung_mask = get_lung_mask(volume, model_name=lungmask_model, force_cpu=force_cpu)
        lung_mask_image = image_from_array_like(lung_mask.astype(np.uint8), full_image)
        lung_mask_image = resample_image(
            lung_mask_image,
            out_spacing=out_spacing,
            interpolator=sitk.sitkNearestNeighbor,
            output_pixel_type=sitk.sitkUInt8,
        )
        lung_mask_array = sitk.GetArrayFromImage(lung_mask_image).astype(bool)
        lung_bbox_zyx = get_lung_bbox(lung_mask_array, padding=lung_padding)
        if lung_bbox_zyx is None:
            print(f"{patient_id}: no lungs detected; writing uncropped volume")
    else:
        lung_mask_array = None

    image = resample_image(image, out_spacing=out_spacing, interpolator=sitk.sitkLinear)
    img_array = lumTrans(sitk.GetArrayFromImage(image))
    extendbox = None

    if crop_lungs and lung_mask_array is not None and lung_bbox_zyx is not None:
        Mask = lung_mask_array
        dilatedMask = process_mask(Mask)
        extendbox = get_extendbox(Mask, margin=lung_padding)
        if extendbox is None:
            print(f"{patient_id}: empty lung mask after resampling; writing uncropped volume")
        else:
            img_array = img_array * dilatedMask
            crop_image_arr = img_array[
                extendbox[0, 0]:extendbox[0, 1],
                extendbox[1, 0]:extendbox[1, 1],
                extendbox[2, 0]:extendbox[2, 1],
            ]
            image_roi = crop_image_to_bbox(image, (extendbox[:, 0], extendbox[:, 1]))
            image = image_from_array_like(crop_image_arr, image_roi)
    else:
        image = image_from_array_like(img_array, image)

    written_mask = None
    label_rows = []
    if write_mask:
        full_shape = sitk.GetArrayFromImage(full_image).shape
        mask_result = generate_nodule_masks(patient_id, full_shape, z_coords, consensus_threshold)
        if mask_result is not None:
            mask_array, instance_masks = mask_result
            mask_image = resample_and_crop_mask_array(mask_array, full_image, out_spacing, extendbox=extendbox)
            instance_mask_images = [
                resample_and_crop_mask_array(instance_mask, full_image, out_spacing, extendbox=extendbox)
                for instance_mask in instance_masks
            ]
            label_rows = labels_from_instance_mask_images(patient_id, instance_mask_images)
            write_nifti(mask_image, mask_path)
            written_mask = str(mask_path)


    write_nifti(image, volume_path)
    save_metadata(output_dir, patient_id, image, lung_bbox_zyx=lung_bbox_zyx, extendbox_zyx=extendbox)
    print(f"{patient_id}: wrote {volume_path}" + (f" and {mask_path}" if written_mask else ""))
    return str(volume_path), written_mask, label_rows


def preprocess_lidc(
    lidc_data,
    savepath,
    consensus_threshold=1,
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
    labels_only=False,
):
    lidc_data = Path(lidc_data)
    savepath = Path(savepath)
    patients = [p for p in lidc_data.iterdir() if p.is_dir() and p.name.startswith("LIDC-IDRI")]
    patients.sort()
    if limit is not None:
        patients = patients[:limit]

    print(f"starting LIDC DICOM preprocessing: {len(patients)} case(s)")
    savepath.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "output_root": str(savepath),
        "consensus_threshold": consensus_threshold,
        "write_mask": write_mask,
        "overwrite": overwrite,
        "crop_lungs": crop_lungs,
        "lung_padding": lung_padding,
        "lungmask_model": lungmask_model,
        "force_cpu": force_cpu,
        "out_spacing": out_spacing,
        "labels_only": labels_only,
    }
    if num_workers > 1:
        worker = partial(process_lidc_case, **kwargs)
        with Pool(num_workers) as pool:
            results = pool.map(worker, [str(patient) for patient in patients])
    else:
        results = [process_lidc_case(str(patient), **kwargs) for patient in patients]

    if write_mask:
        label_rows = []
        for _, _, rows in results:
            label_rows.extend(rows)
        labels_csv = Path(labels_csv) if labels_csv is not None else savepath / "lidc_labels.csv"
        write_boxes_csv(labels_csv, label_rows)
        print(f"wrote labels CSV: {labels_csv}")
    print("end LIDC DICOM preprocessing")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw LIDC-IDRI DICOM CT series to CPMNet-ready NIfTI volumes and nodule masks."
    )
    parser.add_argument("--dicom-dir", "--lidc-data", required=True, help="Root containing LIDC-IDRI-* DICOM folders.")
    parser.add_argument("--output-dir", required=True, help="Directory where per-case NIfTI folders will be written.")
    parser.add_argument("--consensus", type=int, default=1, help="pylidc consensus threshold for nodule masks.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N sorted cases.")
    parser.add_argument("--volume-only", action="store_true", help="Write only CT volumes, not pylidc nodule masks.")
    parser.add_argument("--no-lung-crop", action="store_true", help="Disable LMInferer lung bbox cropping.")
    parser.add_argument("--lung-padding", type=int, default=10, help="Voxel padding around the LMInferer lung bbox.")
    parser.add_argument("--lungmask-model", default="R231", help="LMInferer model name.")
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0), help="Output NIfTI spacing in x y z mm.")
    parser.add_argument("--labels-csv", default=None, help="Single output CSV path for all labels.")
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Reuse existing processed volumes and regenerate masks/labels only.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage for LMInferer.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NIfTI files.")
    args = parser.parse_args()

    preprocess_lidc(
        lidc_data=args.dicom_dir,
        savepath=args.output_dir,
        consensus_threshold=args.consensus,
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
        labels_only=args.labels_only,
    )


if __name__ == "__main__":
    main()
