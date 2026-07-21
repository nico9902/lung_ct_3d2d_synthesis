from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUTS_ROOT = Path("outputs")
DEFAULT_REPORT_DIR = Path("docs/luna16_synthetic_2d_class_rankings")
DEFAULT_DOC_PATH = DEFAULT_REPORT_DIR / "luna16_synthetic_2d_class_rankings.md"
DEFAULT_CSV_PATH = DEFAULT_REPORT_DIR / "luna16_synthetic_2d_class_rankings.csv"
DEFAULT_TOP_BOTTOM_CSV_PATH = DEFAULT_REPORT_DIR / "luna16_synthetic_2d_class_top_bottom.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank synthetic 2D classifier predictions by true-class confidence "
            "and export top/bottom cases per class."
        )
    )
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--ranked-csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--top-bottom-csv", type=Path, default=DEFAULT_TOP_BOTTOM_CSV_PATH)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--include-experiment",
        action="append",
        default=None,
        help="Experiment directory name to include. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.outputs_root, args.include_experiment)
    metrics = load_metrics(args.outputs_root, set(predictions["experiment"].unique()))
    ranked = add_ranking_columns(predictions)
    ranked = add_image_paths(ranked)
    top_bottom = build_top_bottom(ranked, args.top_n)
    best_top_bottom = build_best_backbone_top_bottom(ranked, metrics, args.top_n)

    args.ranked_csv.parent.mkdir(parents=True, exist_ok=True)
    args.top_bottom_csv.parent.mkdir(parents=True, exist_ok=True)
    args.doc_path.parent.mkdir(parents=True, exist_ok=True)

    ranked.to_csv(args.ranked_csv, index=False)
    top_bottom.to_csv(args.top_bottom_csv, index=False)
    write_markdown_report(
        doc_path=args.doc_path,
        ranked=ranked,
        metrics=metrics,
        best_top_bottom=best_top_bottom,
        ranked_csv=args.ranked_csv,
        top_bottom_csv=args.top_bottom_csv,
        top_n=args.top_n,
    )


def load_predictions(outputs_root: Path, include_experiments: list[str] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    allowed = set(include_experiments or [])
    for csv_path in sorted(outputs_root.glob("luna16_synthetic_2d*/all_test_predictions.csv")):
        experiment = csv_path.parent.name
        if allowed and experiment not in allowed:
            continue
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        frame["experiment"] = experiment
        frame["source_predictions_csv"] = str(csv_path)
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No all_test_predictions.csv files found under {outputs_root}")

    predictions = pd.concat(frames, ignore_index=True)
    required = {"experiment", "backbone", "sample_id", "label", "label_name", "prediction", "prediction_name", "score"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction files are missing required columns: {missing}")
    return predictions


def load_metrics(outputs_root: Path, experiments: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for metrics_path in sorted(outputs_root.glob("luna16_synthetic_2d*/prediction_metrics.csv")):
        experiment = metrics_path.parent.name
        if experiment not in experiments:
            continue
        frame = pd.read_csv(metrics_path)
        if frame.empty:
            continue
        frame["experiment"] = experiment
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    metrics = pd.concat(frames, ignore_index=True)
    return metrics[metrics["scope"].astype(str) == "pooled"].copy()


def add_ranking_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    ranked = predictions.copy()
    ranked["score"] = pd.to_numeric(ranked["score"], errors="coerce")
    ranked["label"] = pd.to_numeric(ranked["label"], errors="coerce").astype("Int64")
    ranked["prediction"] = pd.to_numeric(ranked["prediction"], errors="coerce").astype("Int64")
    ranked["correct"] = ranked["label"] == ranked["prediction"]
    ranked["true_class_score"] = ranked.apply(true_class_score, axis=1)
    ranked["error_margin"] = (ranked["score"] - 0.5).abs()
    ranked["predicted_class_score"] = ranked["score"].where(ranked["prediction"] == 1, 1.0 - ranked["score"])
    ranked = ranked.sort_values(
        ["experiment", "backbone", "label_name", "true_class_score", "sample_id"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    ranked["rank_within_true_class"] = (
        ranked.groupby(["experiment", "backbone", "label_name"]).cumcount() + 1
    )
    return ranked


def true_class_score(row: pd.Series) -> float:
    if pd.isna(row["score"]) or pd.isna(row["label"]):
        return float("nan")
    return float(row["score"]) if int(row["label"]) == 1 else 1.0 - float(row["score"])


def add_image_paths(ranked: pd.DataFrame) -> pd.DataFrame:
    ranked = ranked.copy()
    ranked["image_path"] = ranked.apply(
        lambda row: synthetic_image_path(str(row["experiment"]), str(row["sample_id"])),
        axis=1,
    )
    ranked["image_exists"] = ranked["image_path"].map(lambda path: Path(path).exists() if path else False)
    return ranked


def synthetic_image_path(experiment: str, sample_id: str) -> str:
    root = synthetic_root_for_experiment(experiment)
    if root is None:
        return ""
    image_path = root / sample_id / f"surface_{sample_id}.png"
    return str(image_path)


def synthetic_root_for_experiment(experiment: str) -> Path | None:
    prefix = "luna16_synthetic_2d_"
    if not experiment.startswith(prefix):
        return None

    suffix = experiment.removeprefix(prefix)
    if suffix == "gt":
        return Path("data/luna16_saliency_synthetic_gt")
    if suffix == "gt_no_contour":
        return Path("data/luna16_saliency_synthetic_gt_no_contour")
    if suffix == "gt_shepard":
        return Path("data/luna16_saliency_synthetic_gt_shepard")
    if suffix == "half_backbone_gt":
        return Path("data/luna16_saliency_synthetic_gt")
    if suffix == "half_backbone_top5_minprob0.5":
        return Path("data/luna16_saliency_synthetic_detector_top5_minprob0.5")
    if suffix.startswith("top"):
        return Path(f"data/luna16_saliency_synthetic_detector_{suffix}")
    return None


def build_top_bottom(ranked: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    group_columns = ["experiment", "backbone", "label_name"]
    for keys, group in ranked.groupby(group_columns, dropna=False):
        top = group.nlargest(top_n, "true_class_score").copy()
        top["rank_kind"] = "top"
        top["rank_order"] = range(1, len(top) + 1)

        bottom = group.nsmallest(top_n, "true_class_score").copy()
        bottom["rank_kind"] = "bottom"
        bottom["rank_order"] = range(1, len(bottom) + 1)
        rows.extend([top, bottom])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_best_backbone_top_bottom(ranked: pd.DataFrame, metrics: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if metrics.empty:
        best_backbones = ranked[["experiment", "backbone"]].drop_duplicates()
    else:
        sort_columns = ["experiment", "test_mcc", "test_auc", "test_f1", "test_acc", "backbone"]
        best_backbones = (
            metrics.sort_values(sort_columns, ascending=[True, False, False, False, False, True])
            .groupby("experiment", as_index=False)
            .head(1)[["experiment", "backbone"]]
        )
    best_ranked = ranked.merge(best_backbones, on=["experiment", "backbone"], how="inner")
    return build_top_bottom(best_ranked, top_n)


def write_markdown_report(
    doc_path: Path,
    ranked: pd.DataFrame,
    metrics: pd.DataFrame,
    best_top_bottom: pd.DataFrame,
    ranked_csv: Path,
    top_bottom_csv: Path,
    top_n: int,
) -> None:
    lines: list[str] = [
        "# LUNA16 Synthetic 2D Class Rankings",
        "",
        (
            "Questo report ordina le predizioni sintetiche per classe vera usando la confidenza "
            "della classe corretta: `score` per `malignant`, `1 - score` per `benign`."
        ),
        "",
        f"- CSV completo ordinato: `{ranked_csv}`",
        f"- CSV top/bottom per tutti i backbone: `{top_bottom_csv}`",
        f"- Casi mostrati nel report: top {top_n} e bottom {top_n} per classe, sul miglior backbone pooled di ogni esperimento.",
        "- Le immagini puntano ai file `surface_<seriesuid>.png` nelle rispettive directory sintetiche sotto `data/`.",
        "- Il report visualizza `score = P(malignant)` e `class_score = P(classe vera)`.",
        "",
        "## Esperimenti",
        "",
    ]

    summary = experiment_summary(ranked, metrics)
    lines.extend(summary.to_markdown(index=False).splitlines())
    lines.append("")
    lines.extend(interpretation_section(summary))

    for experiment in sorted(best_top_bottom["experiment"].unique()):
        experiment_rows = best_top_bottom[best_top_bottom["experiment"] == experiment]
        if experiment_rows.empty:
            continue
        backbone = str(experiment_rows["backbone"].iloc[0])
        lines.extend(["", f"## {experiment}", "", f"Miglior backbone pooled usato nel report: `{backbone}`.", ""])

        for label_name in sorted(experiment_rows["label_name"].dropna().unique()):
            class_rows = experiment_rows[experiment_rows["label_name"] == label_name]
            lines.extend(["", f"### Classe vera: {label_name}", ""])
            for rank_kind in ["top", "bottom"]:
                section = class_rows[class_rows["rank_kind"] == rank_kind].copy()
                if section.empty:
                    continue
                title = "Top" if rank_kind == "top" else "Bottom"
                lines.extend([f"**{title} {top_n}**", ""])
                table = format_report_table(section, report_dir=doc_path.parent)
                lines.extend(table.to_markdown(index=False).splitlines())
                lines.append("")

    doc_path.write_text("\n".join(lines).rstrip() + "\n")


def interpretation_section(summary: pd.DataFrame) -> list[str]:
    gt = row_by_experiment(summary, "luna16_synthetic_2d_gt")
    top5 = row_by_experiment(summary, "luna16_synthetic_2d_top5_minprob0.5")
    if gt is None or top5 is None:
        return []

    gt_mcc = str(gt.get("best_mcc", ""))
    gt_auc = str(gt.get("best_auc", ""))
    top5_mcc = str(top5.get("best_mcc", ""))
    top5_auc = str(top5.get("best_auc", ""))
    return [
        "",
        "## Lettura GT vs Detector Top5",
        "",
        (
            "Le sintetiche detector-driven `luna16_synthetic_2d_top5_minprob0.5` sono il miglior compromesso "
            "tra le configurazioni detector provate, ma restano sotto le sintetiche ground-truth. "
            f"Nel riepilogo pooled il miglior GT arriva a `MCC {gt_mcc}` e `AUC {gt_auc}`, mentre "
            f"`top5_minprob0.5` arriva a `MCC {top5_mcc}` e `AUC {top5_auc}`."
        ),
        "",
        "La spiegazione piu' plausibile e' che la GT parte da un riferimento anatomico piu' vicino al bersaglio: "
        "la superficie sintetica e' guidata dal nodulo reale, quindi il classificatore riceve un segnale piu' "
        "direttamente correlato alla distinzione benign/malignant.",
        "",
        "Le sintetiche detector-driven dipendono invece dai candidati del detector. Anche quando il detector trova "
        "regioni ragionevoli, i top-k possono includere falsi positivi, localizzazioni non perfettamente centrate "
        "o candidati che aiutano la detection ma non la classificazione della malignita'. In questo senso il top-k "
        "aumenta la copertura, ma puo' diluire il segnale discriminativo del nodulo vero.",
        "",
        "Un altro punto importante e' che il detector ottimizza la localizzazione di candidati nodulo, non la "
        "classificazione benign/malignant. Una regione saliente per detection puo' quindi produrre una superficie "
        "visivamente plausibile ma meno informativa per la label clinica finale.",
        "",
        "I casi bottom del report sono utili per vedere questo limite: quando `class_score` e' vicino a zero, "
        "il classificatore non e' semplicemente incerto, ma spesso e' molto convinto della classe opposta. "
        "I campioni che sono bottom nel detector-driven ma non nella GT sono quindi i casi piu' interessanti "
        "per capire dove il detector sposta, perde o confonde il segnale discriminativo.",
        "",
    ]


def row_by_experiment(summary: pd.DataFrame, experiment: str) -> pd.Series | None:
    rows = summary[summary["experiment"] == experiment]
    if rows.empty:
        return None
    return rows.iloc[0]


def experiment_summary(ranked: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    counts = (
        ranked.groupby("experiment")
        .agg(rows=("sample_id", "size"), backbones=("backbone", "nunique"), samples=("sample_id", "nunique"))
        .reset_index()
    )
    if metrics.empty:
        counts["best_backbone"] = ""
        counts["best_mcc"] = pd.NA
        counts["best_auc"] = pd.NA
        return counts

    best = (
        metrics.sort_values(["experiment", "test_mcc", "test_auc", "test_f1", "test_acc", "backbone"], ascending=[True, False, False, False, False, True])
        .groupby("experiment", as_index=False)
        .head(1)[["experiment", "backbone", "test_mcc", "test_auc"]]
        .rename(columns={"backbone": "best_backbone", "test_mcc": "best_mcc", "test_auc": "best_auc"})
    )
    summary = counts.merge(best, on="experiment", how="left")
    summary["best_mcc"] = summary["best_mcc"].map(format_float)
    summary["best_auc"] = summary["best_auc"].map(format_float)
    return summary


def format_report_table(rows: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    columns = [
        "rank_order",
        "image_path",
        "sample_id",
        "fold",
        "prediction_name",
        "score",
        "true_class_score",
        "correct",
    ]
    table = rows[columns].copy()
    table = table.rename(
        columns={
            "rank_order": "rank",
            "image_path": "image",
            "prediction_name": "pred",
            "true_class_score": "class_score",
        }
    )
    table["image"] = table["image"].map(lambda path: image_html(path, report_dir=report_dir))
    table["score"] = table["score"].map(format_float)
    table["class_score"] = table["class_score"].map(format_float)
    table["correct"] = table["correct"].map(lambda value: "yes" if bool(value) else "no")
    return table


def image_html(path_value: object, report_dir: Path) -> str:
    if pd.isna(path_value) or not str(path_value):
        return ""
    path = Path(str(path_value))
    relative_path = relative_path_from_report(path, report_dir)
    return f'<img src="{relative_path.as_posix()}" width="140">'


def relative_path_from_report(path: Path, report_dir: Path) -> Path:
    try:
        return path.relative_to(report_dir)
    except ValueError:
        return Path("../..") / path


def format_float(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
