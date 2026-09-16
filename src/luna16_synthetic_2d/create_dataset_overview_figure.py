"""Create a dataset overview figure with representative LUNA16/LIDC-IDRI cases.

The figure intentionally shows only the input data and ground-truth nodule
annotations. It does not include detector outputs, adaptive surfaces,
synthetic 2D images, or classifiers.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec, patches
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# Editable configuration
# ---------------------------------------------------------------------------
CONFIG = {
    # Directory containing preprocessed LUNA16 NIfTI volumes and nodule masks.
    # Expected paths match the cv_splits CSV entries, e.g.
    # subset0/<seriesuid>/<seriesuid>_volume.nii.gz.
    "luna16_ct_dir": "data/LUNA16_preprocessed",

    # Metadata file with at least seriesuid, image_path, nodule_mask_path,
    # target_name, max_nodule_mean_malignancy, nodule_count.
    # A LUNA16 classification split CSV is suitable because each fold CSV lists
    # the full cohort and carries the paper's patient-level labels.
    "lidc_metadata_path": "data/LUNA16_preprocessed/cv_splits/luna16_classification_fold0.csv",

    # Optional manual case selection. Leave empty to auto-select suitable cases.
    "benign_patient_id": "1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492",
    "malignant_patient_id": "1.3.6.1.4.1.14519.5.2.1.6279.6001.128023902651233986592378348912",

    # Output stem. The script writes both .png and .pdf.
    "output_path": "docs/figures/dataset_overview_representative_cases",
}


@dataclass(frozen=True)
class NoduleComponent:
    component_id: int
    mask: np.ndarray
    center_zyx: tuple[float, float, float]
    bbox_zyx: tuple[int, int, int, int, int, int]
    voxel_count: int
    approximate_score: float | None


@dataclass(frozen=True)
class CaseData:
    seriesuid: str
    lidc_id: str
    label_name: str
    max_malignancy: float
    volume: np.ndarray
    nodule: NoduleComponent
    row: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--luna16-ct-dir", type=Path, default=Path(CONFIG["luna16_ct_dir"]))
    parser.add_argument("--metadata", type=Path, default=Path(CONFIG["lidc_metadata_path"]))
    parser.add_argument("--benign-patient-id", default=CONFIG["benign_patient_id"])
    parser.add_argument("--malignant-patient-id", default=CONFIG["malignant_patient_id"])
    parser.add_argument("--output", type=Path, default=Path(CONFIG["output_path"]))
    parser.add_argument("--window-center", type=float, default=-600.0)
    parser.add_argument("--window-width", type=float, default=1500.0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_zyx(path: Path) -> np.ndarray:
    arr = read_volume_xyz(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume at {path}, got shape {arr.shape}.")
    return arr.transpose(2, 1, 0)


def read_volume_xyz(path: Path) -> np.ndarray:
    """Read a medical volume as x-y-z array, matching nibabel's get_fdata layout."""
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading NIfTI files requires nibabel. Install the project dependencies "
                "or run this script in the environment used for preprocessing."
            ) from exc
        return nib.load(str(path)).get_fdata().astype(np.float32)

    if suffixes.endswith((".mhd", ".mha")):
        try:
            import SimpleITK as sitk
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading MHD/MHA files requires SimpleITK. Install the project dependencies "
                "or convert the volumes to the preprocessed NIfTI layout."
            ) from exc
        arr_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
        return arr_zyx.transpose(2, 1, 0)

    if suffixes.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 3:
            return arr
        return arr.transpose(2, 1, 0)

    raise ValueError(f"Unsupported volume format: {path}")


def resolve_path(root: Path, raw: object, seriesuid: str, suffix: str) -> Path:
    candidates: list[Path] = []
    raw_text = "" if pd.isna(raw) else str(raw)
    if raw_text:
        raw_path = Path(raw_text)
        candidates.append(raw_path)
        if not raw_path.is_absolute():
            candidates.append(root / raw_path)
            if raw_path.parts and raw_path.parts[0].startswith("subset"):
                candidates.append(root / Path(*raw_path.parts[1:]))

    candidates.append(root / seriesuid / f"{seriesuid}_{suffix}.nii.gz")
    for subset_idx in range(10):
        candidates.append(root / f"subset{subset_idx}" / seriesuid / f"{seriesuid}_{suffix}.nii.gz")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n  ".join(str(p) for p in candidates[:8])
    raise FileNotFoundError(f"Could not resolve {suffix} for {seriesuid}. Tried:\n  {tried}")


def window_ct(image: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(image, low, high)
    return (clipped - low) / (high - low)


def normalize_target_name(row: pd.Series) -> str:
    if "target_name" in row and pd.notna(row["target_name"]):
        return str(row["target_name"]).strip().lower()
    target = row.get("target", "")
    if str(target) in {"1", "True", "true", "malignant"}:
        return "malignant"
    if str(target) in {"0", "False", "false", "benign"}:
        return "benign"
    return "uncertain"


def score_list(row: pd.Series) -> list[float]:
    raw = row.get("nodule_annotation_mean_malignancies", "")
    if pd.isna(raw) or str(raw).strip() in {"", "[]"}:
        return []
    text = str(raw).strip().strip("[]")
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            pass
    return values


def select_case_row(metadata: pd.DataFrame, label: str, requested_id: str) -> pd.Series:
    data = metadata.copy()
    data["label_name_normalized"] = data.apply(normalize_target_name, axis=1)
    data["nodule_count_numeric"] = pd.to_numeric(data.get("nodule_count", 0), errors="coerce").fillna(0)
    data["max_malignancy_numeric"] = pd.to_numeric(
        data.get("max_nodule_mean_malignancy", np.nan), errors="coerce"
    ).fillna(np.nan)

    if requested_id:
        match = data[
            (data["seriesuid"].astype(str) == requested_id)
            | (data.get("lidc_id", pd.Series("", index=data.index)).astype(str) == requested_id)
        ]
        if match.empty:
            raise ValueError(f"Requested {label} patient ID was not found: {requested_id}")
        row = match.iloc[0]
    else:
        if label == "benign":
            candidates = data[
                (data["label_name_normalized"] == "benign")
                & (data["nodule_count_numeric"] > 0)
                & (data["max_malignancy_numeric"] < 3)
            ].copy()
            candidates = candidates.sort_values(["nodule_count_numeric", "max_malignancy_numeric"], ascending=[True, False])
        elif label == "malignant":
            candidates = data[
                (data["label_name_normalized"] == "malignant")
                & (data["nodule_count_numeric"] > 0)
                & (data["max_malignancy_numeric"] > 3)
            ].copy()
            candidates = candidates.sort_values(["max_malignancy_numeric", "nodule_count_numeric"], ascending=[False, True])
        else:
            raise ValueError(f"Unsupported label: {label}")
        if candidates.empty:
            raise ValueError(f"No suitable {label} case found in {metadata}.")
        row = candidates.iloc[0]

    label_name = normalize_target_name(row)
    max_mal = float(pd.to_numeric(row.get("max_nodule_mean_malignancy", np.nan), errors="coerce"))
    if label == "benign" and not (label_name == "benign" and max_mal < 3):
        raise ValueError(f"Selected benign case is inconsistent with max mean malignancy < 3: {row.to_dict()}")
    if label == "malignant" and not (label_name == "malignant" and max_mal > 3):
        raise ValueError(f"Selected malignant case is inconsistent with max mean malignancy > 3: {row.to_dict()}")
    return row


def components_from_mask(mask: np.ndarray, scores: list[float]) -> list[NoduleComponent]:
    labeled, n_components = ndi.label(mask > 0)
    components: list[NoduleComponent] = []
    for component_id in range(1, n_components + 1):
        component = labeled == component_id
        coords = np.argwhere(component)
        if coords.size == 0:
            continue
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)
        center = tuple(float(v) for v in coords.mean(axis=0))
        score = scores[component_id - 1] if component_id - 1 < len(scores) else None
        components.append(
            NoduleComponent(
                component_id=component_id,
                mask=component,
                center_zyx=center,
                bbox_zyx=(int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)),
                voxel_count=int(coords.shape[0]),
                approximate_score=score,
            )
        )
    components.sort(key=lambda c: (c.approximate_score if c.approximate_score is not None else -1, c.voxel_count), reverse=True)
    return components


def load_case(row: pd.Series, root: Path) -> CaseData:
    seriesuid = str(row["seriesuid"])
    volume_path = resolve_path(root, row.get("image_path", ""), seriesuid, "volume")
    mask_path = resolve_path(root, row.get("nodule_mask_path", ""), seriesuid, "nodule_mask")
    volume = load_zyx(volume_path)
    mask = load_zyx(mask_path) > 0
    components = components_from_mask(mask, score_list(row))
    if not components:
        raise ValueError(f"Selected case has no nodule mask components: {seriesuid}")
    return CaseData(
        seriesuid=seriesuid,
        lidc_id=str(row.get("lidc_id", seriesuid)),
        label_name=normalize_target_name(row),
        max_malignancy=float(row.get("max_nodule_mean_malignancy", np.nan)),
        volume=volume,
        nodule=components[0],
        row=row,
    )


def representative_slices(nodule: NoduleComponent, depth: int) -> list[int]:
    z0, z1, _, _, _, _ = nodule.bbox_zyx
    zc = int(round(nodule.center_zyx[0]))
    pad = max(2, int(round((z1 - z0 + 1) * 1.5)))
    return [int(np.clip(zc - pad, 0, depth - 1)), int(np.clip(zc, 0, depth - 1)), int(np.clip(zc + pad, 0, depth - 1))]


def hide_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_mask_contour(ax: plt.Axes, mask_slice: np.ndarray, color: str = "red", linewidth: float = 1.2) -> None:
    if mask_slice.any():
        ax.contour(mask_slice.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)


def crop_bounds(nodule: NoduleComponent, shape_yx: tuple[int, int], margin: int = 18) -> tuple[int, int, int, int]:
    _, _, y0, y1, x0, x1 = nodule.bbox_zyx
    h, w = shape_yx
    side = max(y1 - y0 + 1, x1 - x0 + 1) + 2 * margin
    side = max(side, 48)
    cy = int(round((y0 + y1) / 2))
    cx = int(round((x0 + x1) / 2))
    half = side // 2
    return cy - half, cy + half, cx - half, cx + half


def crop_with_padding(image: np.ndarray, y0: int, y1: int, x0: int, x1: int, fill: float = 0.0) -> np.ndarray:
    out = np.full((y1 - y0, x1 - x0), fill, dtype=image.dtype)
    h, w = image.shape
    sy0, sy1 = max(y0, 0), min(y1, h)
    sx0, sx1 = max(x0, 0), min(x1, w)
    dy0, dx0 = sy0 - y0, sx0 - x0
    out[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = image[sy0:sy1, sx0:sx1]
    return out


def draw_volume_stack(ax: plt.Axes, case: CaseData, window_center: float, window_width: float) -> None:
    volume = case.volume
    depth, height, width = volume.shape
    zc, yc, xc = case.nodule.center_zyx
    # z_samples = np.linspace(max(0, zc - depth * 0.25), min(depth - 1, zc + depth * 0.25), 7).astype(int)
    zc = int(round(zc))

    offsets = [18, 12, 6, 0, -6, -12, -18]

    z_samples = [
        int(np.clip(zc + off, 0, depth - 1))
        for off in offsets
    ]

    thumb_w = 1.0
    thumb_h = height / width
    dx = 0.08
    dy = 0.06
    for idx, z in enumerate(z_samples):
        x0 = idx * dx
        y0 = idx * dy
        extent = [x0, x0 + thumb_w, y0, y0 + thumb_h]
        ax.imshow(volume[z], cmap="gray", origin="upper", extent=extent, zorder=idx)
        ax.add_patch(patches.Rectangle((x0, y0), thumb_w, thumb_h, fill=False, edgecolor="0.20", linewidth=0.5, zorder=idx + 0.1))

        if z == zc:
            px = x0 + (xc / max(width - 1, 1)) * thumb_w
            py = y0 + thumb_h - (yc / max(height - 1, 1)) * thumb_h

            ax.add_patch(
                patches.Circle(
                    (px, py),
                    0.045,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.2,
                    zorder=idx + 1
                )
            )

    ax.text(0.04, thumb_h + len(z_samples) * dy + 0.03, "axial slice stack", fontsize=7, ha="left", va="bottom", color="0.25")
    ax.set_xlim(-0.03, thumb_w + dx * (len(z_samples) - 1) + 0.08)
    ax.set_ylim(-0.03, thumb_h + dy * (len(z_samples) - 1) + 0.12)
    hide_axis(ax)


def draw_slice_panel(
    ax: plt.Axes,
    case: CaseData,
    z: int,
    window_center: float,
    window_width: float,
    show_z_label: bool = True,
) -> None:
    ax.imshow(case.volume[z], cmap="gray", origin="upper")
    draw_mask_contour(ax, case.nodule.mask[z])
    if show_z_label:
        ax.text(0.03, 0.95, f"z={z}", transform=ax.transAxes, fontsize=7, color="white", ha="left", va="top")
    hide_axis(ax)


def draw_zoom_panel(ax: plt.Axes, case: CaseData, window_center: float, window_width: float) -> None:
    z = int(round(case.nodule.center_zyx[0]))
    y0, y1, x0, x1 = crop_bounds(case.nodule, case.volume.shape[1:])
    crop = crop_with_padding(case.volume[z], y0, y1, x0, x1, fill=window_center - window_width / 2)
    mask_crop = crop_with_padding(case.nodule.mask[z].astype(np.float32), y0, y1, x0, x1, fill=0) > 0
    ax.imshow(crop, cmap="gray", origin="upper")
    draw_mask_contour(ax, mask_crop, linewidth=1.5)
    if case.nodule.approximate_score is not None:
        label = f"mean mal. {case.nodule.approximate_score:.2g}"
    else:
        label = f"max mean mal. {case.max_malignancy:.2g}"
    ax.text(0.04, 0.95, label, transform=ax.transAxes, fontsize=7, color="white", ha="left", va="top")
    hide_axis(ax)


def add_case_label(fig: plt.Figure, ax: plt.Axes, case: CaseData) -> None:
    bbox = ax.get_position()
    label = f"{case.label_name.capitalize()} patient"
    fig.text(bbox.x0 - 0.045, bbox.y0 + bbox.height * 0.50, label, ha="right", va="center", fontsize=8)


def create_figure(cases: list[CaseData], output: Path, window_center: float, window_width: float, dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.2, 3.6), facecolor="white")
    outer = gridspec.GridSpec(
        nrows=3,
        ncols=3,
        figure=fig,
        height_ratios=[0.12, 1.0, 1.0],
        width_ratios=[1.15, 2.05, 0.95],
        hspace=0.12,
        wspace=0.10,
        left=0.19,
        right=0.985,
        top=0.93,
        bottom=0.06,
    )

    headers = ["(a) 3D CT volume", "(b) Representative axial slices", "(c) Nodule close-up"]
    for col, header in enumerate(headers):
        ax = fig.add_subplot(outer[0, col])
        ax.text(0.5, 0.35, header, ha="center", va="center", fontsize=8, fontweight="bold")
        hide_axis(ax)

    first_axes = []
    for row_idx, case in enumerate(cases, start=1):
        ax_stack = fig.add_subplot(outer[row_idx, 0])
        draw_volume_stack(ax_stack, case, window_center, window_width)
        first_axes.append(ax_stack)

        slice_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[row_idx, 1], wspace=0.03)
        for idx, z in enumerate(representative_slices(case.nodule, case.volume.shape[0])):
            ax_slice = fig.add_subplot(slice_grid[0, idx])
            draw_slice_panel(ax_slice, case, z, window_center, window_width)

        ax_zoom = fig.add_subplot(outer[row_idx, 2])
        draw_zoom_panel(ax_zoom, case, window_center, window_width)

    for ax, case in zip(first_axes, cases):
        add_case_label(fig, ax, case)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    benign_row = select_case_row(metadata, "benign", args.benign_patient_id)
    malignant_row = select_case_row(metadata, "malignant", args.malignant_patient_id)
    cases = [load_case(benign_row, args.luna16_ct_dir), load_case(malignant_row, args.luna16_ct_dir)]
    create_figure(cases, args.output, args.window_center, args.window_width, args.dpi)
    for case in cases:
        print(
            f"{case.label_name}: {case.lidc_id} / {case.seriesuid}, "
            f"max mean malignancy={case.max_malignancy:.3g}, "
            f"selected z={case.nodule.center_zyx[0]:.1f}"
        )
    print(f"Wrote {args.output.with_suffix('.png')} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
