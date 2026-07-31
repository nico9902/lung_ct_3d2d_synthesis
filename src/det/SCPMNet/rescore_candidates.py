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

from src.det.SCPMNet.fp_reduction import CandidatePatchDataset, FPReductionLitModel, evaluate_froc, ground_truth_by_series


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore SCPMNet candidates with a trained FP-reduction classifier.")
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--csv-path", default="data/lidc_process/lidc_labels.csv")
    parser.add_argument("--data-root", default="data/lidc_process")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", default="outputs/scpmnet/fp_reduction/rescored_test")
    parser.add_argument("--patch-size", type=int, nargs=3, default=(32, 32, 32))
    parser.add_argument("--clip", type=float, nargs=2, default=(-1000.0, 400.0))
    parser.add_argument(
        "--intensity-mode",
        choices=("hu", "uint8", "auto"),
        default="hu",
        help="Patch intensity normalization. Must match FPR training.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--volume-cache-size", type=int, default=4)
    parser.add_argument(
        "--normalized-volume-cache-dir",
        default=None,
        help="Optional directory with normalized .npy volumes to avoid repeatedly decoding NIfTI files.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-mode", choices=("classifier", "multiply", "average"), default="multiply")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates)
    if "label" not in candidates.columns:
        tmp = output_dir / "_unlabeled_candidates_for_rescore.csv"
        candidates.assign(label=0, ignore=False).to_csv(tmp, index=False)
        candidates_csv = tmp
    else:
        candidates_csv = Path(args.candidates)

    dataset = CandidatePatchDataset(
        candidates_csv=candidates_csv,
        csv_path=args.csv_path,
        split=args.split,
        data_root=args.data_root,
        patch_size=args.patch_size,
        clip=args.clip,
        intensity_mode=args.intensity_mode,
        include_ignored=True,
        augment=False,
        volume_cache_size=args.volume_cache_size,
        normalized_volume_cache_dir=args.normalized_volume_cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    model = FPReductionLitModel.load_from_checkpoint(args.classifier_checkpoint)
    model.eval().to(args.device)
    scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Rescoring candidates"):
            logits = model(batch["image"].to(args.device), batch["meta"].to(args.device))
            scores.extend(torch.sigmoid(logits).detach().cpu().tolist())

    rescored = candidates.copy().iloc[: len(scores)].reset_index(drop=True)
    if "scpm_probability" not in rescored.columns:
        rescored["scpm_probability"] = rescored["probability"].astype(float)
    rescored["classifier_probability"] = scores
    if args.score_mode == "classifier":
        rescored["probability"] = rescored["classifier_probability"]
    elif args.score_mode == "average":
        rescored["probability"] = 0.5 * rescored["scpm_probability"].astype(float) + 0.5 * rescored["classifier_probability"].astype(float)
    else:
        rescored["probability"] = rescored["scpm_probability"].astype(float) * rescored["classifier_probability"].astype(float)

    prediction_cols = ["seriesuid", "coordZ", "coordY", "coordX", "radius", "probability", "scpm_probability", "classifier_probability"]
    predictions_path = output_dir / f"{args.split}_predictions_rescored.csv"
    rescored[prediction_cols].to_csv(predictions_path, index=False)

    gt = ground_truth_by_series(args.csv_path, args.split, args.data_root, skip_missing_images=False)
    froc, curve, mean_froc = evaluate_froc(rescored, gt, score_col="probability")
    froc_path = output_dir / f"{args.split}_froc_rescored.csv"
    curve_path = output_dir / f"{args.split}_froc_curve_rescored.csv"
    froc.to_csv(froc_path, index=False)
    curve.to_csv(curve_path, index=False)
    print(f"Wrote rescored predictions to {predictions_path}")
    print(f"Wrote FROC to {froc_path}; mean={mean_froc:.4f}")


if __name__ == "__main__":
    main()
