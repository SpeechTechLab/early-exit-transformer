#!/usr/bin/env python3
"""Build DDK comparison tables (F1, accuracy, ROC-AUC) from classification run summaries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "classification_results"

LANG_ORDER = ["CZ", "DE", "ES"]
LANG_LATEX = {"CZ": "Czech (CZ)", "DE": "German (DE)", "ES": "Colombian (CO)"}
LANG_LABEL = {"CZ": "cz", "DE": "de", "ES": "co"}
MODEL_ORDER = [("rf", "Random Forest"), ("xgb", "XGBoost")]
LATEX_COLUMNS = [
    "H", "W", "W-FT", "W-CTC", "G", "H+G", "W+G", "W-FT+G", "W-CTC+G", "H+W-FT+G",
]

# IEEE two-column papers: wide tables must use table* + \textwidth, not table + \columnwidth.
LATEX_TABLE_BEGIN = "\\begin{table*}[!t]"
LATEX_TABLE_END = "\\end{table*}"
LATEX_TABLE_FONT = "\\footnotesize"
LATEX_RESIZE_BEGIN = r"\resizebox{\textwidth}{!}{%"


@dataclass(frozen=True)
class ColumnSpec:
    label: str
    run_dir: str
    feature_subset: str
    csv_match: Optional[str] = None
    csv_exclude: Optional[str] = None


COLUMNS = [
    ColumnSpec("HuBERT", "run_20260520_122815", "hubert_all"),
    ColumnSpec(
        "Whisper",
        "run_20260604_112910",
        "whisper_all",
        csv_match="_whisper_DDK.csv",
    ),
    ColumnSpec(
        "Whisper (fine-tuned)",
        "run_20260605_100410",
        "whisper_all",
        csv_match="whisper_ft_DDK",
        csv_exclude="glottal",
    ),
    ColumnSpec(
        "Whisper (CTC fine-tuned)",
        "run_whisper_ctc_ddk",
        "whisper_all",
        csv_match="whisper_ctc_DDK",
        csv_exclude="glottal",
    ),
    ColumnSpec(
        "Glottal (QCP)",
        "run_20260515_100930",
        "glottal_plus_direct",
        csv_match="QCP_python_DDK",
    ),
    ColumnSpec(
        "HuBERT + glottal",
        "run_hubert_glottal_combined",
        "hubert_plus_glottal_plus_direct",
        csv_match="hubert_glottal_DDK",
    ),
    ColumnSpec(
        "Whisper + glottal",
        "run_20260604_112910",
        "whisper_plus_glottal_plus_direct",
        csv_match="whisper_glottal_DDK",
    ),
    ColumnSpec(
        "Whisper (fine-tuned) + glottal",
        "run_20260605_100410",
        "whisper_plus_glottal_plus_direct",
        csv_match="whisper_ft_glottal_DDK",
    ),
    ColumnSpec(
        "Whisper (CTC fine-tuned) + glottal",
        "run_whisper_ctc_ddk",
        "whisper_plus_glottal_plus_direct",
        csv_match="whisper_ctc_glottal_DDK",
    ),
    ColumnSpec(
        "HuBERT + Whisper + glottal",
        "run_20260607_130745",
        "hubert_plus_whisper_plus_glottal_plus_direct",
        csv_match="hubert_whisper_ft_glottal_DDK",
    ),
]

METRIC_CHOICES = {
    "sample": ("sample_f1_mean", "sample_f1_std", "Sample-level F1 (0.5 threshold)"),
    "paper": ("paper_f1_mean", "paper_f1_std", "Speaker-level F1 (sens ≥ 0.90)"),
}

# (level tag, short name, mean column, std column)
DUAL_LEVEL_METRICS: dict[str, tuple[str, str, str, str]] = {
    "f1": ("Spk.", "F1", "paper_f1_mean", "paper_f1_std"),
    "acc_spk": ("Spk.", "Acc", "speaker_accuracy_mean", "speaker_accuracy_std"),
    "auc_spk": ("Spk.", "AUC", "speaker_auc_mean", "speaker_auc_std"),
    "f1_utt": ("Utt.", "F1", "sample_f1_mean", "sample_f1_std"),
    "acc_utt": ("Utt.", "Acc", "sample_accuracy_mean", "sample_accuracy_std"),
    "auc_utt": ("Utt.", "AUC", "sample_auc_mean", "sample_auc_std"),
}

DUAL_METRIC_ORDER = ["f1", "acc_spk", "auc_spk", "f1_utt", "acc_utt", "auc_utt"]

MULTI_METRIC_CHOICES = {
    "f1": ("sample_f1_mean", "sample_f1_std", "F1 (utterance-level, threshold 0.5)"),
    "acc": ("sample_accuracy_mean", "sample_accuracy_std", "Accuracy (utterance-level, threshold 0.5)"),
    "auc": ("sample_auc_mean", "sample_auc_std", "ROC-AUC (utterance-level)"),
}


def _load_run(run_dir: str) -> Optional[pd.DataFrame]:
    path = RESULTS_ROOT / run_dir / "global_summary_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _pick_row(df: pd.DataFrame, spec: ColumnSpec, language: str, model: str) -> Optional[pd.Series]:
    mask = (
        (df["language"] == language)
        & (df["model"] == model)
        & (df["feature_subset"] == spec.feature_subset)
        & (df["csv_file"].str.contains("DDK", case=False, na=False))
    )
    if spec.csv_match:
        mask &= df["csv_file"].str.contains(spec.csv_match, case=False, na=False)
    if spec.csv_exclude:
        mask &= ~df["csv_file"].str.contains(spec.csv_exclude, case=False, na=False)
    rows = df[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f}±{std:.3f}"


def _bold_if_best(text: str, mean: float, best_mean: float) -> str:
    if mean >= best_mean - 1e-9:
        return f"**{text}**"
    return text


def _latex_cell(mean: Optional[float], std: Optional[float], bold: bool) -> str:
    if mean is None or std is None:
        return "---"
    body = f"{mean:.3f}$\\pm${std:.3f}"
    return f"\\textbf{{{body}}}" if bold else body


def build_table(language: str, metric: str) -> tuple[list[str], list[list[str]]]:
    mean_col, std_col, _ = METRIC_CHOICES[metric]
    header = ["Classifier"] + [c.label for c in COLUMNS]
    rows: list[list[str]] = []

    for model_id, model_name in MODEL_ORDER:
        cells = [model_name]
        means: list[Optional[float]] = []
        formatted: list[str] = []

        for spec in COLUMNS:
            df = _load_run(spec.run_dir)
            if df is None:
                means.append(None)
                formatted.append("---")
                continue
            row = _pick_row(df, spec, language, model_id)
            if row is None or pd.isna(row.get(mean_col)):
                means.append(None)
                formatted.append("---")
                continue
            mean_v = float(row[mean_col])
            std_v = float(row[std_col])
            means.append(mean_v)
            formatted.append(_fmt(mean_v, std_v))

        valid_means = [m for m in means if m is not None]
        best = max(valid_means) if valid_means else float("-inf")
        for i, text in enumerate(formatted):
            if means[i] is not None:
                formatted[i] = _bold_if_best(text, means[i], best)
        rows.append([model_name] + formatted)

    return header, rows


def _row_metric_cells(
    language: str,
    model_id: str,
    mean_col: str,
    std_col: str,
) -> tuple[list[Optional[float]], list[Optional[float]]]:
    means: list[Optional[float]] = []
    stds: list[Optional[float]] = []
    for spec in COLUMNS:
        df = _load_run(spec.run_dir)
        if df is None:
            means.append(None)
            stds.append(None)
            continue
        row = _pick_row(df, spec, language, model_id)
        if row is None or mean_col not in row or pd.isna(row.get(mean_col)):
            means.append(None)
            stds.append(None)
            continue
        means.append(float(row[mean_col]))
        std_v = row.get(std_col)
        stds.append(float(std_v) if std_v is not None and not pd.isna(std_v) else 0.0)
    return means, stds


def build_latex_table(language: str, metric: str) -> str:
    _, _, metric_desc = METRIC_CHOICES[metric]
    label = f"tab:ddk_{LANG_LABEL[language]}"
    title = LANG_LATEX[language]
    ncol = len(LATEX_COLUMNS)
    lines = [
        f"% {title} — {metric_desc}",
        LATEX_TABLE_BEGIN,
        f"\\caption{{DDK classification on {title}. {metric_desc} ($\\mu \\pm \\sigma$). Best per row in bold.}}",
        f"\\label{{{label}}}",
        "\\centering",
        LATEX_TABLE_FONT,
        LATEX_RESIZE_BEGIN,
        f"\\begin{{tabular}}{{|l|c|{'c|' * ncol}}}",
        "\\hline",
        "\\textbf{Classifier} & "
        + " & ".join(f"\\textbf{{{c}}}" for c in LATEX_COLUMNS)
        + " \\\\",
        "\\hline",
    ]

    for model_id, model_name in MODEL_ORDER:
        mean_col, std_col, _ = METRIC_CHOICES[metric]
        means, stds = _row_metric_cells(language, model_id, mean_col, std_col)
        valid = [m for m in means if m is not None]
        best = max(valid) if valid else float("-inf")
        cells = [
            _latex_cell(means[i], stds[i], means[i] is not None and means[i] >= best - 1e-9)
            for i in range(len(COLUMNS))
        ]
        lines.append(f"{model_name} & " + " & ".join(cells) + " \\\\")

    lines.extend(["\\hline", "\\end{tabular}}", LATEX_TABLE_END, ""])
    return "\n".join(lines)


def build_latex_dual_table(language: str, metric_keys: list[str] | None = None) -> str:
    """Speaker-level (Spk.) and utterance-level (Utt.) rows per classifier and metric."""
    if metric_keys is None:
        metric_keys = list(DUAL_METRIC_ORDER)

    label = f"tab:ddk_{LANG_LABEL[language]}"
    title = LANG_LATEX[language]
    ncol = len(LATEX_COLUMNS)
    n_rows_per_clf = len(metric_keys)
    lines = [
        f"% {title} — dual evaluation (F1 / Acc / AUC)",
        LATEX_TABLE_BEGIN,
        f"\\caption{{DDK classification on {title} ($\\mu \\pm \\sigma$; best per row in bold). "
        f"\\emph{{Spk.}}: speaker-aggregated scores; F1 with threshold tuned for sens.\\ $\\geq 0.90$; "
        f"Acc at fixed $0.5$; AUC from speaker probabilities. "
        f"\\emph{{Utt.}}: per-utterance predictions (threshold $0.5$).}}",
        f"\\label{{{label}}}",
        "\\centering",
        LATEX_TABLE_FONT,
        LATEX_RESIZE_BEGIN,
        f"\\begin{{tabular}}{{|l|l|l|{'c|' * ncol}}}",
        "\\hline",
        "\\textbf{Clf.} & \\textbf{Lvl.} & \\textbf{Metr.} & "
        + " & ".join(f"\\textbf{{{c}}}" for c in LATEX_COLUMNS)
        + " \\\\",
        "\\hline",
    ]

    clf_short = {"Random Forest": "RF", "XGBoost": "XGB"}
    for model_id, model_name in MODEL_ORDER:
        clf_label = clf_short[model_name]
        for row_idx, key in enumerate(metric_keys):
            level_tag, metr_tag, mean_col, std_col = DUAL_LEVEL_METRICS[key]
            means, stds = _row_metric_cells(language, model_id, mean_col, std_col)
            valid = [m for m in means if m is not None]
            best = max(valid) if valid else float("-inf")
            cells = [
                _latex_cell(means[i], stds[i], means[i] is not None and means[i] >= best - 1e-9)
                for i in range(len(COLUMNS))
            ]
            if row_idx == 0:
                prefix = f"\\multirow{{{n_rows_per_clf}}}{{*}}{{{clf_label}}} & {level_tag} & {metr_tag} & "
            else:
                prefix = f" & {level_tag} & {metr_tag} & "
            lines.append(prefix + " & ".join(cells) + " \\\\")
        lines.append("\\hline")

    lines.extend(["\\end{tabular}}", LATEX_TABLE_END, ""])
    return "\n".join(lines)


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_dual_table(language: str, metric_keys: list[str] | None = None) -> tuple[list[str], list[list[str]]]:
    if metric_keys is None:
        metric_keys = list(DUAL_METRIC_ORDER)
    header = ["Classifier", "Level", "Metric"] + [c.label for c in COLUMNS]
    rows: list[list[str]] = []
    for model_id, model_name in MODEL_ORDER:
        for key in metric_keys:
            level_tag, metr_tag, mean_col, std_col = DUAL_LEVEL_METRICS[key]
            means, stds = _row_metric_cells(language, model_id, mean_col, std_col)
            valid = [m for m in means if m is not None]
            best = max(valid) if valid else float("-inf")
            cells = []
            for i in range(len(COLUMNS)):
                if means[i] is None:
                    cells.append("---")
                else:
                    text = _fmt(means[i], stds[i])
                    cells.append(_bold_if_best(text, means[i], best))
            rows.append([model_name, level_tag, metr_tag] + cells)
    return header, rows


def render_markdown(metric: str, dual: bool = False) -> str:
    lang_titles = {"CZ": "Czech (CZ)", "DE": "German (DE)", "ES": "Colombian (ES)"}
    if dual:
        metric_line = (
            "**Metrics:** F1, accuracy, ROC-AUC; **Spk.** = speaker-aggregated "
            "(F1: sens. ≥ 0.90 threshold; Acc: 0.5); **Utt.** = utterance-level (0.5).  "
        )
    else:
        _, _, metric_desc = METRIC_CHOICES[metric]
        metric_line = f"**Metric:** {metric_desc} (mean ± std over 10-fold outer CV).  "
    parts = [
        "# DDK classification — results by classifier and feature setup",
        "",
        "**Task:** OneVoice DDK only (German, Czech, Colombian).  ",
        metric_line,
        "**Models:** `facebook/hubert-base-ls960`, `openai/whisper-large-v3` (encoder, mean-pooled). "
        "Whisper finetune = Bridge2AI read-speech `checkpoint-150`.",
        "",
        "**Source runs**",
        "",
        "| Column | Run directory | Feature subset | CSV filter |",
        "|--------|---------------|----------------|--------------|",
    ]
    for spec in COLUMNS:
        filt = spec.csv_match or "(any DDK csv)"
        if spec.csv_exclude:
            filt += f", exclude {spec.csv_exclude!r}"
        parts.append(
            f"| {spec.label} | `{spec.run_dir}` | `{spec.feature_subset}` | `{filt}` |"
        )
    parts.append("")
    regen = (
        "Regenerate: `python3 build_ddk_f1_tables.py --write-md --write-latex --dual-level`"
        if dual
        else "Regenerate: `python3 build_ddk_f1_tables.py --metric sample --write-md --write-latex`"
    )
    parts.append(regen)
    parts.append("")

    for lang in LANG_ORDER:
        if dual:
            header, rows = build_dual_table(lang)
        else:
            header, rows = build_table(lang, metric)
        parts.extend(["---", "", f"## {lang_titles[lang]}", ""])
        parts.append(_markdown_table(header, rows))
        parts.append("")

    parts.extend(
        [
            "## Notes",
            "",
            "- **Glottal (QCP)** = QCP inverse-filtering features only (`glottal_plus_direct`, 63 dims; run `run_20260515_100930`).",
            "- **+ glottal** in fusion columns = same QCP feature set concatenated with SSL embeddings.",
            "- **Whisper (fine-tuned):** Bridge2AI English read-speech ASR finetune (~8.4% test WER).",
            "- **HuBERT + Whisper + glottal:** uses **finetuned** Whisper + HuBERT + glottal (`--triple_ft_defaults`).",
            "- `---` means the run directory is missing locally; run `scripts/pull_cluster_classification_results.sh`.",
            "",
        ]
    )
    return "\n".join(parts)


def render_latex(metric: str, dual: bool = False) -> str:
    parts = [
        "% Auto-generated by build_ddk_f1_tables.py — paste into Experiments and Results",
        "% Wide tables use table* + \\textwidth for IEEE two-column layout.",
        "% Preamble (if missing): \\usepackage{multirow} \\usepackage{dblfloatfix}",
        "",
        r"Tables~\ref{tab:ddk_cz}--\ref{tab:ddk_co} report DDK F1, accuracy, and ROC-AUC "
        r"(mean $\pm$ std over 10-fold nested group cross-validation). "
        r"Abbreviations: H=HuBERT, W=Whisper, G=glottal (QCP), FT=fine-tuned on Bridge2AI read speech; "
        r"H+G, W+G, etc.\ denote concatenation with the same glottal feature set.",
        "",
    ]
    if dual:
        for lang in LANG_ORDER:
            parts.append(build_latex_dual_table(lang))
    else:
        _, _, metric_desc = METRIC_CHOICES[metric]
        parts.insert(2, "% Metric: " + metric_desc)
        for lang in LANG_ORDER:
            parts.append(build_latex_table(lang, metric))
    parts.extend(
        [
            r"Glottal-only performance is competitive on German (RF Utt.\ F1: 0.752) but weak on Czech. "
            r"Fusion with SSL embeddings generally outperforms G alone; the best configuration remains language- and evaluation-level dependent. "
            r"Columns marked `---` for W-FT or H+W-FT+G indicate incomplete local result snapshots; run "
            r"\texttt{scripts/pull\_cluster\_classification\_results.sh} for full Acc/AUC.",
            "",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_CHOICES),
        default="sample",
        help="sample = sample_f1 (0.5 threshold); paper = speaker-level paper_f1",
    )
    parser.add_argument(
        "--write-md",
        action="store_true",
        help="Write classification_results/ddk_classifier_f1_tables.md",
    )
    parser.add_argument(
        "--write-latex",
        action="store_true",
        help="Write classification_results/ddk_paper_snippets.tex",
    )
    parser.add_argument(
        "--dual-level",
        action="store_true",
        help="LaTeX tables with Spk./Utt. rows (use with --write-latex)",
    )
    args = parser.parse_args()

    if not args.write_md and not args.write_latex:
        text = render_markdown(args.metric, dual=args.dual_level)
        print(text)
    if args.write_md:
        out = RESULTS_ROOT / "ddk_classifier_f1_tables.md"
        out.write_text(render_markdown(args.metric, dual=args.dual_level), encoding="utf-8")
        print(f"Wrote {out}", flush=True)
    if args.write_latex:
        out = RESULTS_ROOT / "ddk_paper_snippets.tex"
        out.write_text(render_latex(args.metric, dual=args.dual_level), encoding="utf-8")
        print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
