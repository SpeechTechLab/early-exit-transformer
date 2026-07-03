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
    "H", "W", "W-FT", "W-CTC", "G", "H+G", "W+G", "W-FT+G", "W-CTC+G", "H+W+G", "H+W-FT+G",
    "LF-H+W+G", "LF-H+W-FT+G", "LF-H+W-CTC+G",
]

FUSION_COMPARE_LATEX = ["H+W+G", "H+W-FT+G", "LF-H+W+G", "LF-H+W-FT+G"]

DECIMAL_PLACES = 2

LANG_COLOR = {"CZ": "red", "DE": "purple", "ES": "blue"}
LANG_HEADER = {"CZ": "CZECH", "DE": "GERMAN", "ES": "SPANISH"}

# Utterance-level summary tables (column label, COLUMNS index)
SUMMARY_SINGLE_COLS = [("H", 0), ("W", 1), ("G", 4)]
SUMMARY_HW_G_COLS = [("H", 0), ("W", 1), ("H+G", 5), ("W+G", 6)]
SUMMARY_FUSION_COLS = [("H+W+G", 9), ("LF-H+W+G", 11)]
SUMMARY_WHISPER_FT_COLS = [("W", 1), ("W-CE", 2), ("W-CTC", 3)]

SUMMARY_METRIC_ROWS = [
    ("F1", "sample_f1_mean", "sample_f1_std"),
    ("Acc", "sample_accuracy_mean", "sample_accuracy_std"),
    ("AUC", "sample_auc_mean", "sample_auc_std"),
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
        "whisper_ctc_ddk",
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
        "whisper_ctc_ddk",
        "whisper_plus_glottal_plus_direct",
        csv_match="whisper_ctc_glottal_DDK",
    ),
    ColumnSpec(
        "HuBERT + Whisper + glottal (frozen Whisper)",
        "run_hwg_ddk",
        "hubert_plus_whisper_plus_glottal_plus_direct",
        csv_match="hubert_whisper_glottal_DDK",
        csv_exclude="whisper_ft",
    ),
    ColumnSpec(
        "HuBERT + Whisper-FT + glottal",
        "run_20260607_130745",
        "hubert_plus_whisper_plus_glottal_plus_direct",
        csv_match="hubert_whisper_ft_glottal_DDK",
    ),
    ColumnSpec("LF-H+W+G", "run_late_fusion_ddk", "LF-H+W+G"),
    ColumnSpec("LF-H+W-FT+G", "run_late_fusion_ddk", "LF-H+W-FT+G"),
    ColumnSpec("LF-H+W-CTC+G", "run_late_fusion_ddk", "LF-H+W-CTC+G"),
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


def _num_fmt(value: float) -> str:
    return f"{value:.{DECIMAL_PLACES}f}"


def _fmt(mean: float, std: float) -> str:
    return f"{_num_fmt(mean)}±{_num_fmt(std)}"


def _bold_if_best(text: str, mean: float, best_mean: float) -> str:
    if mean >= best_mean - 1e-9:
        return f"**{text}**"
    return text


def _latex_cell(mean: Optional[float], std: Optional[float], bold: bool) -> str:
    if mean is None or std is None:
        return "---"
    body = f"{_num_fmt(mean)}$\\pm${_num_fmt(std)}"
    return f"\\textbf{{{body}}}" if bold else body


def _latex_summary_cell(
    mean: Optional[float],
    std: Optional[float],
    bold: bool,
    lang: Optional[str] = None,
    emph: bool = False,
) -> str:
    if mean is None or std is None:
        return "---"
    m = _num_fmt(mean)
    s = _num_fmt(std)
    if bold and lang:
        body = f"\\textbf{{\\color{{{LANG_COLOR[lang]}}}{{{m}}}}}$\\pm${s}"
    elif bold:
        body = f"\\textbf{{{m}$\\pm${s}}}"
    else:
        body = f"{m}$\\pm${s}"
    return f"\\emph{{{body}}}" if emph else body


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


def _column_indices(labels: list[str]) -> list[int]:
    return [i for i, spec in enumerate(COLUMNS) if LATEX_COLUMNS[i] in labels]


def build_latex_fusion_compare_table(language: str) -> str:
    """Early vs late triple fusion: H+W+G, H+W-FT+G, LF-H+W+G, LF-H+W-FT+G."""
    col_idxs = _column_indices(FUSION_COMPARE_LATEX)
    label = f"tab:ddk_fusion_{LANG_LABEL[language]}"
    title = LANG_LATEX[language]
    ncol = len(col_idxs)
    n_rows_per_clf = len(DUAL_METRIC_ORDER)
    lines = [
        f"% {title} — early vs late triple fusion",
        LATEX_TABLE_BEGIN,
        f"\\caption{{DDK triple fusion on {title}: early concatenation vs.\\ late fusion "
        f"($\\mu \\pm \\sigma$; best per row in bold). "
        f"\\emph{{Spk.}} F1 uses sens.\\ $\\geq 0.90$; \\emph{{Utt.}} uses threshold $0.5$.}}",
        f"\\label{{{label}}}",
        "\\centering",
        LATEX_TABLE_FONT,
        LATEX_RESIZE_BEGIN,
        f"\\begin{{tabular}}{{|l|l|l|{'c|' * ncol}}}",
        "\\hline",
        "\\textbf{Clf.} & \\textbf{Lvl.} & \\textbf{Metr.} & "
        + " & ".join(f"\\textbf{{{FUSION_COMPARE_LATEX[i]}}}" for i in range(ncol))
        + " \\\\",
        "\\hline",
    ]
    clf_short = {"Random Forest": "RF", "XGBoost": "XGB"}
    for model_id, model_name in MODEL_ORDER:
        clf_label = clf_short[model_name]
        for row_idx, key in enumerate(DUAL_METRIC_ORDER):
            level_tag, metr_tag, mean_col, std_col = DUAL_LEVEL_METRICS[key]
            all_means, all_stds = _row_metric_cells(language, model_id, mean_col, std_col)
            means = [all_means[i] for i in col_idxs]
            stds = [all_stds[i] for i in col_idxs]
            valid = [m for m in means if m is not None]
            best = max(valid) if valid else float("-inf")
            cells = [
                _latex_cell(means[i], stds[i], means[i] is not None and means[i] >= best - 1e-9)
                for i in range(ncol)
            ]
            if row_idx == 0:
                prefix = f"\\multirow{{{n_rows_per_clf}}}{{*}}{{{clf_label}}} & {level_tag} & {metr_tag} & "
            else:
                prefix = f" & {level_tag} & {metr_tag} & "
            lines.append(prefix + " & ".join(cells) + " \\\\")
        lines.append("\\hline")
    lines.extend(["\\end{tabular}}", LATEX_TABLE_END, ""])
    return "\n".join(lines)


def render_latex_fusion_compare() -> str:
    parts = [
        "% Auto-generated by build_ddk_f1_tables.py --fusion-compare",
        "% Preamble: \\usepackage{multirow} \\usepackage{dblfloatfix}",
        "",
        r"Tables~\ref{tab:ddk_fusion_cz}--\ref{tab:ddk_fusion_co} compare \textbf{early} triple fusion "
        r"(feature concatenation: H+W+G with frozen Whisper; H+W-FT+G with CE fine-tuned Whisper) "
        r"against \textbf{late} fusion (LF-H+W+G, LF-H+W-FT+G; weighted modality posteriors, $\sum w=1$).",
        "",
    ]
    for lang in LANG_ORDER:
        parts.append(build_latex_fusion_compare_table(lang))
    parts.extend(
        [
            r"Late fusion improves speaker-level F1 over early concatenation on German and Colombian Spanish; "
            r"on Czech, LF-H+W-FT+G matches or slightly exceeds H+W+G. "
            r"H+W-FT+G remains competitive when Whisper is CE-adapted on Bridge2AI read speech.",
            "",
        ]
    )
    return "\n".join(parts)


def _summary_lang_header(ncol: int) -> list[str]:
    parts: list[str] = []
    for i, lang in enumerate(LANG_ORDER):
        color = LANG_COLOR[lang]
        sep = "c|" if i == len(LANG_ORDER) - 1 else "c||"
        parts.append(
            f"\\multicolumn{{{ncol}}}{{{sep}}}{{\\textbf{{\\color{{{color}}}{{{LANG_HEADER[lang]}}}}}}}"
        )
    return parts


def _build_summary_table_rows(
    metr_tag: str,
    mean_col: str,
    std_col: str,
    col_specs: list[tuple[str, int]],
) -> list[str]:
    """Return RF, XGB, and AVG rows for one metric."""
    ncol = len(col_specs)
    per_clf_cells: list[list[str]] = []
    per_clf_nums: list[list[tuple[Optional[float], Optional[float]]]] = []

    for clf_id, _ in MODEL_ORDER:
        row_cells: list[str] = []
        nums: list[tuple[Optional[float], Optional[float]]] = []
        for lang in LANG_ORDER:
            means: list[Optional[float]] = []
            stds: list[Optional[float]] = []
            for _, col_idx in col_specs:
                m_list, s_list = _row_metric_cells(lang, clf_id, mean_col, std_col)
                means.append(m_list[col_idx])
                stds.append(s_list[col_idx])
            valid = [m for m in means if m is not None]
            best = max(valid) if valid else float("-inf")
            for m, s in zip(means, stds):
                row_cells.append(
                    _latex_summary_cell(m, s, m is not None and m >= best - 1e-9, lang)
                )
                nums.append((m, s))
        per_clf_cells.append(row_cells)
        per_clf_nums.append(nums)

    avg_cells: list[str] = []
    for lang_idx, lang in enumerate(LANG_ORDER):
        col_avgs: list[tuple[float, float]] = []
        for col_i in range(ncol):
            nums_m: list[float] = []
            nums_s: list[float] = []
            for clf_nums in per_clf_nums:
                m, s = clf_nums[lang_idx * ncol + col_i]
                if m is not None and s is not None:
                    nums_m.append(m)
                    nums_s.append(s)
            if nums_m:
                col_avgs.append((sum(nums_m) / len(nums_m), sum(nums_s) / len(nums_s)))
            else:
                col_avgs.append((float("nan"), float("nan")))

        valid_avgs = [m for m, _ in col_avgs if m == m]
        best_avg = max(valid_avgs) if valid_avgs else float("-inf")
        for avg_m, avg_s in col_avgs:
            if avg_m != avg_m:
                avg_cells.append("---")
            else:
                avg_cells.append(
                    _latex_summary_cell(
                        avg_m, avg_s, avg_m >= best_avg - 1e-9, lang, emph=True
                    )
                )

    return [
        f"RF  & {metr_tag}  & " + " & ".join(per_clf_cells[0]) + " \\\\",
        f"XGB & {metr_tag}  & " + " & ".join(per_clf_cells[1]) + " \\\\",
        f"\\emph{{AVG}} & \\emph{{{metr_tag}}} & " + " & ".join(avg_cells) + " \\\\",
    ]


def _render_summary_multilang_table(
    label: str,
    caption: str,
    col_specs: list[tuple[str, int]],
    col_headers: list[str] | None = None,
    resize: bool = True,
) -> str:
    nlang = len(LANG_ORDER)
    ncol_per_lang = len(col_specs)
    total_cols = nlang * ncol_per_lang
    col_labels = col_headers or [name for name, _ in col_specs]

    lines = [
        f"% Summary: {label}",
        LATEX_TABLE_BEGIN,
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        "\\centering",
        LATEX_TABLE_FONT,
    ]
    if resize:
        lines.append(LATEX_RESIZE_BEGIN)

    tab_cols = "|l|l|" + "c|" * total_cols
    lines.append(f"\\begin{{tabular}}{{{tab_cols}}}")

    # Header row 1: language groups
    cline_end = 2 + total_cols
    lines.append(f"\\cline{{3-{cline_end}}}")
    lang_hdr = " & ".join(_summary_lang_header(ncol_per_lang))
    lines.append(f"\\multicolumn{{2}}{{c|}}{{}} & {lang_hdr} \\\\")

    lines.append("\\hline")
    col_hdr = " & ".join(col_labels * nlang)
    lines.append(f"\\textbf{{Clf.}} & \\textbf{{Metr.}} & {col_hdr} \\\\")
    lines.append("\\hline")

    metric_lines: list[str] = []
    for i, (metr_tag, mean_col, std_col) in enumerate(SUMMARY_METRIC_ROWS):
        if i > 0:
            metric_lines.append("\\hline")
        metric_lines.extend(_build_summary_table_rows(metr_tag, mean_col, std_col, col_specs))

    lines.extend(metric_lines)
    lines.append("\\hline")
    if resize:
        lines.extend(["\\end{tabular}}", LATEX_TABLE_END, ""])
    else:
        lines.extend(["\\end{tabular}", LATEX_TABLE_END, ""])
    return "\n".join(lines)


def render_latex_summary_tables() -> str:
    cap_single = (
        "Performance on DDK with individual features families. Results are reported as "
        "($\\mu \\pm \\sigma$; best in bold). {\\em AVG}: average across classifiers."
    )
    cap_hw = (
        "Performance on DDK using HuBERT/Whisper embeddings and their concatenation with "
        "glottal features. Results are reported as ($\\mu \\pm \\sigma$; best in bold). "
        "{\\em AVG}: average across classifiers."
    )
    cap_fusion = (
        "Performance on DDK obtained with early- and late-fusion strategies for combining "
        "classifier outputs. Results are reported as ($\\mu \\pm \\sigma$; best in bold). "
        "{\\em AVG}: average across classifiers."
    )
    cap_wft = (
        "Performance on DDK with Whisper fine-tuning. Results are reported as "
        "($\\mu \\pm \\sigma$; best in bold). {\\em AVG} average across classifiers."
    )

    parts = [
        "% Auto-generated by build_ddk_f1_tables.py --summary-tables",
        f"% Utterance-level metrics; {DECIMAL_PLACES} decimal places.",
        "",
        _render_summary_multilang_table("ddk_single_all", cap_single, SUMMARY_SINGLE_COLS),
        _render_summary_multilang_table("ddk_hw_hg_wg", cap_hw, SUMMARY_HW_G_COLS),
    ]

    # Early vs late fusion table (custom column headers per language)
    nlang = len(LANG_ORDER)
    fusion_lines = [
        "% Summary: early vs late triple fusion",
        LATEX_TABLE_BEGIN,
        f"\\caption{{{cap_fusion}}}",
        "\\label{tab:ddk_early_late_fusion}",
        "\\centering",
        LATEX_TABLE_FONT,
        "\\begin{tabular}{|l|l|c|c||c|c||c|c|}",
        "\\cline{3-8}",
        "\\multicolumn{2}{c|}{} & "
        + " & ".join(_summary_lang_header(2))
        + " \\\\",
        "\\multicolumn{2}{c|}{} & "
        + " & ".join(
            h
            for _ in LANG_ORDER
            for h in ("\\textbf{early}", "\\textbf{late}")
        )
        + " \\\\",
        "\\hline",
        "\\textbf{Clf.} & \\textbf{Metr.} & "
        + " & ".join(
            f"\\textbf{{{name}}}"
            for _ in LANG_ORDER
            for name in ("H+W+G", "LF-H+W+G")
        )
        + " \\\\",
        "\\hline",
    ]
    for i, (metr_tag, mean_col, std_col) in enumerate(SUMMARY_METRIC_ROWS):
        if i > 0:
            fusion_lines.append("\\hline")
        fusion_lines.extend(
            _build_summary_table_rows(metr_tag, mean_col, std_col, SUMMARY_FUSION_COLS)
        )
    fusion_lines.extend(["\\hline", "\\end{tabular}", LATEX_TABLE_END, ""])
    parts.append("\n".join(fusion_lines))

    parts.append(
        _render_summary_multilang_table(
            "ddk_W_tuned",
            cap_wft,
            SUMMARY_WHISPER_FT_COLS,
            col_headers=["W", "W-CE", "W-CTC"],
        )
    )
    return "\n".join(parts)


def render_latex(metric: str, dual: bool = False, fusion_compare: bool = False) -> str:
    if fusion_compare:
        return render_latex_fusion_compare()
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
            r"Glottal-only performance is competitive on German (RF Utt.\ F1: 0.75) but weak on Czech. "
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
    parser.add_argument(
        "--fusion-compare",
        action="store_true",
        help="Write classification_results/ddk_fusion_compare.tex (H+W+G vs LF columns only)",
    )
    parser.add_argument(
        "--summary-tables",
        action="store_true",
        help="Write classification_results/ddk_paper_summary_tables.tex",
    )
    args = parser.parse_args()

    if not args.write_md and not args.write_latex and not args.summary_tables:
        text = render_markdown(args.metric, dual=args.dual_level)
        print(text)
    if args.write_md:
        out = RESULTS_ROOT / "ddk_classifier_f1_tables.md"
        out.write_text(render_markdown(args.metric, dual=args.dual_level), encoding="utf-8")
        print(f"Wrote {out}", flush=True)
    if args.write_latex:
        if args.fusion_compare:
            out = RESULTS_ROOT / "ddk_fusion_compare.tex"
            out.write_text(render_latex(args.metric, dual=args.dual_level, fusion_compare=True), encoding="utf-8")
            print(f"Wrote {out}", flush=True)
        else:
            out = RESULTS_ROOT / "ddk_paper_snippets.tex"
            out.write_text(render_latex(args.metric, dual=args.dual_level), encoding="utf-8")
            print(f"Wrote {out}", flush=True)
    if args.summary_tables:
        out = RESULTS_ROOT / "ddk_paper_summary_tables.tex"
        out.write_text(render_latex_summary_tables(), encoding="utf-8")
        print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
