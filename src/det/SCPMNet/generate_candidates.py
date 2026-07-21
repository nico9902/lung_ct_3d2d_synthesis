from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.det.SCPMNet.dataset import SCPMSlidingWindowDataset, scpm_sliding_collate
from src.det.SCPMNet.fp_reduction import ground_truth_by_series, label_candidates
from src.det.SCPMNet.lightning_model import SCPMLitModel, sphere_nms


def generate_candidates(args: argparse.Namespace) -> pd.DataFrame:
    dataset = SCPMSlidingWindowDataset(
        csv_path=args.csv_path,
        split=args.split,
        data_root=args.data_root,
        crop_size=tuple(args.crop_size),
        stride=tuple(args.sliding_window_stride),
        clip=tuple(args.clip),
        skip_missing_images=not args.keep_missing_images,
    )
    if len(dataset) == 0:
        raise ValueError(f"No windows found for split={args.split!r}.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=scpm_sliding_collate,
        pin_memory=args.device.startswith("cuda"),
    )
    model = SCPMLitModel.load_from_checkpoint(
        args.checkpoint,
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
        decode_threshold=args.decode_threshold,
        decode_topk=args.decode_topk,
        nms_threshold=args.nms_threshold,
        final_topk=args.final_topk,
        evaluate_froc=False,
    )
    model.eval().to(args.device)
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Generating {args.split} candidates"):
            images = batch["image"].to(args.device)
            outputs = model(images)
            for i, seriesuid in enumerate(batch["seriesuid"]):
                detections = model.decode_one(outputs, i).detach().cpu()
                if len(detections):
                    detections[:, :3] += batch["origin"][i].detach().cpu().view(1, 3)
                for z, y, x, radius, score in detections.tolist():
                    rows.append([seriesuid, z, y, x, radius, score])

    columns = ["seriesuid", "coordZ", "coordY", "coordX", "radius", "probability"]
    merged_rows = []
    if rows:
        raw_df = pd.DataFrame(rows, columns=columns)
        for seriesuid, group in raw_df.groupby("seriesuid", sort=False):
            detections = torch.as_tensor(
                group[["coordZ", "coordY", "coordX", "radius", "probability"]].to_numpy(),
                dtype=torch.float32,
            )
            detections = sphere_nms(detections, args.nms_threshold, args.final_topk)
            for rank, (z, y, x, radius, score) in enumerate(detections.tolist(), start=1):
                merged_rows.append([seriesuid, z, y, x, radius, score, rank])
    candidates = pd.DataFrame(merged_rows, columns=[*columns, "candidate_rank"])
    if args.label_candidates:
        gt = ground_truth_by_series(args.csv_path, args.split, args.data_root, skip_missing_images=not args.keep_missing_images)
        candidates = label_candidates(candidates, gt, ignore_margin=args.ignore_margin)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SCPMNet candidates for any split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv-path", default="data/lidc_process/lidc_labels.csv")
    parser.add_argument("--data-root", default="data/lidc_process")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crop-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--sliding-window-stride", type=int, nargs=3, default=(24, 24, 24))
    parser.add_argument("--clip", type=float, nargs=2, default=(-1000.0, 400.0))
    parser.add_argument("--decode-threshold", type=float, default=0.05)
    parser.add_argument("--decode-topk", type=int, default=300)
    parser.add_argument("--final-topk", type=int, default=300)
    parser.add_argument(
        "--top-candidates-per-volume",
        type=int,
        default=None,
        help="Semantic alias for --final-topk. Use 100 for Top-100-per-volume FPR experiments.",
    )
    parser.add_argument("--nms-threshold", type=float, default=0.05)
    parser.add_argument("--label-candidates", action="store_true")
    parser.add_argument("--ignore-margin", type=float, default=2.0)
    parser.add_argument("--keep-missing-images", action="store_true")
    args = parser.parse_args()
    if args.top_candidates_per_volume is not None:
        args.final_topk = int(args.top_candidates_per_volume)

    candidates = generate_candidates(args)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_csv, index=False)
    print(f"Wrote {len(candidates)} candidates to {output_csv}")
    if "label" in candidates.columns:
        usable = candidates[~candidates["ignore"].astype(bool)] if "ignore" in candidates.columns else candidates
        print(usable["label"].value_counts().rename(index={0: "negative", 1: "positive"}).to_string())


if __name__ == "__main__":
    main()
