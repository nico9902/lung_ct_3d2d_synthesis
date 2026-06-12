"""Visualize a CPMNetv2 LIDC training crop with GT center and 3D box.

The plotted annotation is the exact post-transform training annotation:
image shape is [1, D, H, W] and boxes are [z, y, x, d, h, w, class].
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cpmnetv2_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.det.CPMNetv2.dataload.dataset_lidc import LIDCCPMNetTrainDataset
from src.det.CPMNetv2.transform.label import CoordToAnnot
from src.det.CPMNetv2.transform.rotate import RandomTranspose


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", default="data/dataset_nodule_mean.csv")
    parser.add_argument("--images-dir", default="data/lidc_process")
    parser.add_argument("--annotations-dir", default="data/lidc_process")
    parser.add_argument("--labels-csv", default="data/lidc_process/lidc_labels.csv")
    parser.add_argument("--case", default=None, help="Optional case id, e.g. LIDC-IDRI-0002")
    parser.add_argument("--view", default="axial", choices=("axial", "coronal", "sagittal"))
    parser.add_argument("--crop-size", nargs=3, type=int, default=(64, 128, 128), metavar=("D", "H", "W"))
    parser.add_argument("--spacing", nargs=3, type=float, default=(0.7, 0.3125, 0.3125), metavar=("Z", "Y", "X"))
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=64, help="How many random crops/cases to try for a positive sample")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--annot-index", type=int, default=0, help="Which foreground annotation to center slices on")
    parser.add_argument("--force-transpose", choices=("none", "xy", "zy", "zx"), default="none")
    parser.add_argument("--output", default="outputs/cpmnetv2/debug/training_sample_overlay.png")
    return parser.parse_args()


def train_cases(csv_path: Path, labels_csv: Path, case: str | None):
    if case:
        return [case]

    split_df = pd.read_csv(csv_path)
    labels_df = pd.read_csv(labels_csv)
    positive_cases = set(labels_df.loc[labels_df["label"].astype(str) == "nodule", "seriesuid"])
    cases = split_df.loc[split_df["split"] == "train", "patient_id"].drop_duplicates().tolist()
    return [case_id for case_id in cases if case_id in positive_cases]


def find_positive_crop(args):
    cases = train_cases(Path(args.csv_path), Path(args.labels_csv), args.case)
    if not cases:
        raise RuntimeError("No train cases with nodule labels were found.")

    for attempt in range(args.attempts):
        case_id = cases[attempt % len(cases)]
        image_path = Path(args.images_dir) / case_id / f"{case_id}_volume.nii.gz"
        if not image_path.exists():
            continue

        dataset = LIDCCPMNetTrainDataset(
            images_dir=args.images_dir,
            annotations_dir=args.annotations_dir,
            case_list=[case_id],
            view=args.view,
            crop_size=args.crop_size,
            spacing=args.spacing,
            num_samples=args.num_samples,
            csv_file=args.labels_csv,
        )
        for sample_idx, sample in enumerate(dataset[0]):
            annot = sample["annot"]
            foreground = annot[annot[:, -1] >= 0]
            if foreground.size > 0:
                return case_id, sample_idx, sample, foreground

    raise RuntimeError(f"No positive crop found after {args.attempts} attempts. Try increasing --attempts.")


def force_transpose(sample: dict, mode: str):
    if mode == "none":
        return sample

    kwargs = {
        "xy": {"trans_xy": True, "trans_zy": False, "trans_zx": False},
        "zy": {"trans_xy": False, "trans_zy": True, "trans_zx": False},
        "zx": {"trans_xy": False, "trans_zy": False, "trans_zx": True},
    }[mode]
    sample = RandomTranspose(p=1.0, **kwargs)(sample)
    return CoordToAnnot(blank_side=0)(sample)


def robust_window(image: np.ndarray):
    lo, hi = np.percentile(image, (1, 99))
    if np.isclose(lo, hi):
        lo, hi = float(image.min()), float(image.max())
    return lo, hi


def add_box(ax, lower_left, width, height, color):
    ax.add_patch(Rectangle(lower_left, width, height, fill=False, edgecolor=color, linewidth=1.8))


def draw_cross(ax, x, y, color):
    ax.scatter([x], [y], marker="+", s=120, color=color, linewidths=2.0)


def plot_sample(case_id: str, sample_idx: int, sample: dict, foreground: np.ndarray, annot_index: int, output: Path):
    image = np.asarray(sample["image"][0])
    depth, height, width = image.shape
    annot_index = min(max(annot_index, 0), len(foreground) - 1)
    center_box = foreground[annot_index]
    z, y, x, d, h, w, _ = center_box
    zi = int(np.clip(round(z), 0, depth - 1))
    yi = int(np.clip(round(y), 0, height - 1))
    xi = int(np.clip(round(x), 0, width - 1))

    vmin, vmax = robust_window(image)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    planes = [
        ("axial z", axes[0], image[zi, :, :], "x", "y"),
        ("coronal y", axes[1], image[:, yi, :], "x", "z"),
        ("sagittal x", axes[2], image[:, :, xi], "y", "z"),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(foreground), 1)))
    for title, ax, plane, xlabel, ylabel in planes:
        ax.imshow(plane, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    for idx, box in enumerate(foreground):
        bz, by, bx, bd, bh, bw, _ = box
        color = colors[idx]
        add_box(axes[0], (bx - bw / 2, by - bh / 2), bw, bh, color)
        draw_cross(axes[0], bx, by, color)
        add_box(axes[1], (bx - bw / 2, bz - bd / 2), bw, bd, color)
        draw_cross(axes[1], bx, bz, color)
        add_box(axes[2], (by - bh / 2, bz - bd / 2), bh, bd, color)
        draw_cross(axes[2], by, bz, color)

    fig.suptitle(
        f"{case_id} sample={sample_idx} crop={tuple(image.shape)} "
        f"center(z,y,x)=({z:.1f},{y:.1f},{x:.1f}) box(d,h,w)=({d:.1f},{h:.1f},{w:.1f})"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    return fig, center_box


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    case_id, sample_idx, sample, foreground = find_positive_crop(args)
    before_box = foreground[args.annot_index].copy()
    sample = force_transpose(sample, args.force_transpose)
    foreground = sample["annot"][sample["annot"][:, -1] >= 0]
    output = Path(args.output)
    fig, center_box = plot_sample(case_id, sample_idx, sample, foreground, args.annot_index, output)

    print(f"Saved: {output}")
    print(f"Case: {case_id}")
    print(f"Sample index: {sample_idx}")
    print(f"Image shape [D,H,W]: {tuple(sample['image'][0].shape)}")
    print(f"Foreground boxes: {len(foreground)}")
    print(f"Forced transpose: {args.force_transpose}")
    if args.force_transpose != "none":
        print("Before forced transpose [z,y,x,d,h,w,class]: " + np.array2string(before_box, precision=2))
    print("Selected annot [z,y,x,d,h,w,class]: " + np.array2string(center_box, precision=2))
    plt.close(fig)


if __name__ == "__main__":
    main()
