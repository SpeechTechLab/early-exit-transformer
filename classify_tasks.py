import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)
from typing import List, Optional, Tuple, Union
from xgboost import XGBClassifier


# ------------------------------
# Hardcoded run configuration
# Choose one or more from: 'svm', 'rf', 'xgb'
# ------------------------------
RUN_MODELS = ['rf', 'xgb']

def aggregate_mean_by_group(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate probabilities per cougher/group by mean. Cougher label is majority vote."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    g = np.asarray(g)
    uniq = np.unique(g)
    y_g = np.zeros(len(uniq), dtype=int)
    p_g = np.zeros(len(uniq), dtype=float)

    for i, gg in enumerate(uniq):
        idx = np.where(g == gg)[0]
        p_g[i] = float(np.mean(p[idx])) if len(idx) else float('nan')
        y_g[i] = int(np.mean(y[idx]) >= 0.5) if len(idx) else 0

    return y_g, p_g, uniq


def summarize_metrics(title: str, results_sample: dict, results_speaker: dict):
    print(f"\n{title}")
    print(f"{'Metric':<14} | {'Sample-Level':<20} | {'Speaker-Level':<20}")
    print("-" * 64)

    for metric in results_sample.keys():
        mean_samp = np.nanmean(results_sample[metric])
        std_samp = np.nanstd(results_sample[metric])

        valid_spk = [v for v in results_speaker[metric] if not np.isnan(v)]
        mean_spk = np.mean(valid_spk) if valid_spk else np.nan
        std_spk = np.std(valid_spk) if valid_spk else np.nan

        print(f"{metric:<14} | {mean_samp:.4f} ± {std_samp:.4f}      | {mean_spk:.4f} ± {std_spk:.4f}")


def _sens_spec_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return (sensitivity, specificity) for binary labels where positive class is 1."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return float(sens), float(spec)


def _select_threshold_max_spec_at_min_sens(
    y_true: np.ndarray, y_score: np.ndarray, min_sens: float
) -> float:
    """
    Select threshold that maximizes specificity subject to sensitivity >= min_sens.
    If no threshold meets the constraint, returns 1.0 (predict all negatives).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    uniq = np.unique(y_score[~np.isnan(y_score)])
    if uniq.size == 0:
        return 0.5
    # Consider thresholds from high -> low (more conservative -> more sensitive)
    thresholds = np.r_[uniq, 1.0]
    best_thr = 1.0
    best_spec = -1.0
    for thr in sorted(thresholds, reverse=True):
        y_pred = (y_score >= thr).astype(int)
        sens, spec = _sens_spec_from_labels(y_true, y_pred)
        if np.isnan(sens) or np.isnan(spec):
            continue
        if sens >= float(min_sens) and spec > best_spec:
            best_spec = spec
            best_thr = float(thr)
    return float(best_thr)


def compute_metrics_summary_row(results_sample: dict, results_speaker: dict):
    row = {}
    for metric in results_sample.keys():
        row[f"sample_{metric}_mean"] = float(np.nanmean(results_sample[metric]))
        row[f"sample_{metric}_std"] = float(np.nanstd(results_sample[metric]))
        row[f"speaker_{metric}_mean"] = float(np.nanmean(results_speaker[metric]))
        row[f"speaker_{metric}_std"] = float(np.nanstd(results_speaker[metric]))
    return row


def compute_paper_metrics_summary_row(fold_df: pd.DataFrame) -> dict:
    """Aggregate per-fold paper-style speaker metrics (sens/spec/F1) into summary columns."""
    row = {}
    if fold_df is None or fold_df.empty:
        return row
    mapping = {
        "sensitivity": "paper_speaker_sensitivity",
        "specificity": "paper_speaker_specificity",
        "f1": "paper_speaker_f1",
    }
    for short_name, col in mapping.items():
        if col not in fold_df.columns:
            continue
        vals = fold_df[col].astype(float)
        row[f"paper_{short_name}_mean"] = float(np.nanmean(vals))
        row[f"paper_{short_name}_std"] = float(np.nanstd(vals))
    if "paper_min_sens" in fold_df.columns:
        row["paper_min_sens"] = float(fold_df["paper_min_sens"].iloc[0])
    return row


def infer_language_code(csv_file: str) -> str:
    """Map feature CSV filename to paper language code (CZ, DE, ES, CO)."""
    stem = Path(csv_file).stem.lower()
    if "german" in stem:
        return "DE"
    if "czech" in stem:
        return "CZ"
    if "colombian" in stem:
        return "ES"
    if "vowels" in stem:
        return "CO"
    return "UNK"


def is_dedicated_vowels_csv(csv_file: str) -> bool:
    """
    True for vowels-only feature tables (e.g. features_Vowels_QCP_python.csv)
    where task names do not contain 'vowel' (e.g. Glottal_signals_db).
    """
    stem = Path(csv_file).stem.lower()
    if "ddk" in stem:
        return False
    if any(lang in stem for lang in ("german", "czech", "colombian")):
        return False
    return "vowels" in stem


def infer_task_type(csv_file: str) -> str:
    """DDK, vowels, or other (from filename and active task filter)."""
    stem = Path(csv_file).stem.lower()
    if "ddk" in stem:
        return "DDK"
    if "vowels" in stem or is_dedicated_vowels_csv(csv_file):
        return "Vowels"
    task_contains = str(globals().get("TASK_CONTAINS", "") or "").lower()
    if "vowel" in task_contains:
        return "Vowels"
    return "Other"


LANGUAGE_NAMES = {
    "CZ": "Czech",
    "DE": "German",
    "ES": "Colombian (DDK)",
    "CO": "Colombian (Vowels)",
    "UNK": "Unknown",
}

TASK_SECTION_ORDER = ["DDK", "Vowels", "Other"]
LANG_ORDER = ["CZ", "DE", "ES", "CO", "UNK"]

# Hernandez et al., arXiv:2603.22225v2 — HuBERT-Large, oral DDK, speaker-level metrics (mean, std).
# Table 3 monolingual (Mono.): train/test on same target language — closest to our per-language setup.
PAPER_BASELINE_TABLE3_MONO_HUBERT = {
    "CZ": {
        "specificity": (0.53, 0.12),
        "sensitivity": (0.89, 0.07),
        "f1": (0.76, 0.05),
    },
    "DE": {
        "specificity": (0.47, 0.06),
        "sensitivity": (0.91, 0.03),
        "f1": (0.75, 0.02),
    },
    "ES": {
        "specificity": (0.41, 0.09),
        "sensitivity": (0.88, 0.06),
        "f1": (0.71, 0.04),
    },
}

def _fmt_mean_std(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "—"
    if np.isnan(std):
        return f"{mean:.2f}"
    return f"{mean:.2f}±{std:.2f}"


def _paper_metric_columns() -> list:
    return [
        ("Specificity", "paper_specificity_mean", "paper_specificity_std"),
        ("Sensitivity", "paper_sensitivity_mean", "paper_sensitivity_std"),
        ("F1", "paper_f1_mean", "paper_f1_std"),
    ]


def _prepare_paper_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize global summary into one row per (task, language, model, feature_subset)."""
    if summary_df.empty:
        return summary_df

    df = summary_df.copy()
    if "language" not in df.columns:
        df["language"] = df["csv_file"].map(infer_language_code)
    if "task_type" not in df.columns:
        df["task_type"] = df["csv_file"].map(infer_task_type)

    rename = {
        "paper_sensitivity_mean": "paper_sensitivity_mean",
        "paper_specificity_mean": "paper_specificity_mean",
        "paper_f1_mean": "paper_f1_mean",
        "paper_sensitivity_std": "paper_sensitivity_std",
        "paper_specificity_std": "paper_specificity_std",
        "paper_f1_std": "paper_f1_std",
    }
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    required = [
        "paper_sensitivity_mean",
        "paper_specificity_mean",
        "paper_f1_mean",
    ]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    return df


def print_paper_style_language_table(
    subset_df: pd.DataFrame,
    title: str,
    min_sens: float,
):
    """Print one paper-style table: rows = languages, columns = Spec / Sens / F1 (mean±std)."""
    if subset_df.empty:
        return

    lang_order = ["CZ", "DE", "ES", "CO", "UNK"]
    langs_present = [l for l in lang_order if l in subset_df["language"].values]
    langs_present += sorted(set(subset_df["language"]) - set(langs_present))

    print(f"\n{title}")
    print(f"(speaker-level, threshold tuned for sensitivity ≥ {min_sens:.2f})")
    header = f"{'Lang.':<6} | {'Specificity':<16} | {'Sensitivity':<16} | {'F1':<16}"
    print(header)
    print("-" * len(header))

    for lang in langs_present:
        row = subset_df[subset_df["language"] == lang]
        if row.empty:
            continue
        r = row.iloc[0]
        spec = _fmt_mean_std(r.get("paper_specificity_mean", np.nan), r.get("paper_specificity_std", np.nan))
        sens = _fmt_mean_std(r.get("paper_sensitivity_mean", np.nan), r.get("paper_sensitivity_std", np.nan))
        f1 = _fmt_mean_std(r.get("paper_f1_mean", np.nan), r.get("paper_f1_std", np.nan))
        print(f"{lang:<6} | {spec:<16} | {sens:<16} | {f1:<16}")


def build_paper_summary_from_folds(folds_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild per-language paper summary rows from per-fold metrics."""
    if folds_df.empty:
        return pd.DataFrame()

    group_cols = ["csv_file", "task_stem", "model", "feature_subset"]
    missing = [c for c in group_cols if c not in folds_df.columns]
    if missing:
        return pd.DataFrame()

    rows = []
    for keys, g in folds_df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["language"] = infer_language_code(row["csv_file"])
        row["task_type"] = infer_task_type(row["csv_file"])
        for short_name, col in [
            ("sensitivity", "paper_speaker_sensitivity"),
            ("specificity", "paper_speaker_specificity"),
            ("f1", "paper_speaker_f1"),
        ]:
            if col not in g.columns:
                continue
            vals = g[col].astype(float)
            row[f"paper_{short_name}_mean"] = float(np.nanmean(vals))
            row[f"paper_{short_name}_std"] = float(np.nanstd(vals))
        if "paper_min_sens" in g.columns:
            row["paper_min_sens"] = float(g["paper_min_sens"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _paper_baseline_cell(lang: str, metric: str) -> str:
    """Table 3 monolingual HuBERT-Large baseline (Hernandez et al., arXiv:2603.22225v2)."""
    row = PAPER_BASELINE_TABLE3_MONO_HUBERT.get(lang)
    if not row:
        return "—"
    mean, std = row[metric]
    return _fmt_mean_std(mean, std)


def select_best_combination_per_language(paper_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (task_type, language), pick the model + feature_subset with highest paper F1.
    Ties break on specificity, then sensitivity.
    """
    if paper_df.empty:
        return paper_df

    df = paper_df.copy()
    sort_cols = ["paper_f1_mean", "paper_specificity_mean", "paper_sensitivity_mean"]
    for c in sort_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df.sort_values(
        ["task_type", "language"] + sort_cols,
        ascending=[True, True, False, False, False],
    )
    return df.groupby(["task_type", "language"], as_index=False).first()


def write_best_results_markdown(
    paper_df: pd.DataFrame,
    run_output_dir: Path,
    min_sens: float,
) -> Path:
    """Write best_results_by_language.md — best model/features per language for DDK and Vowels."""
    best_df = select_best_combination_per_language(paper_df)
    if best_df.empty:
        return None

    best_df = best_df.copy()
    best_df["language_name"] = best_df["language"].map(lambda c: LANGUAGE_NAMES.get(c, c))
    best_df.to_csv(run_output_dir / "best_results_by_language.csv", index=False)

    run_name = run_output_dir.name
    lines = [
        "# Best classification results per language",
        "",
        "Speaker-level metrics (**mean ± std** over outer CV folds). "
        f"Threshold tuned for **sensitivity ≥ {min_sens:.2f}** "
        "(OneVoice paper protocol, [arXiv:2603.22225v2](https://arxiv.org/abs/2603.22225)).",
        "",
        f"For each language, the **best** model and feature subset is chosen by highest **F1** "
        "(ties: specificity, then sensitivity).",
        "",
        f"**Run:** `{run_output_dir}`",
        "",
    ]

    task_types = [t for t in TASK_SECTION_ORDER if t in best_df["task_type"].values]
    task_types += sorted(set(best_df["task_type"]) - set(task_types))

    for task_type in task_types:
        section = best_df[best_df["task_type"] == task_type]
        if section.empty:
            continue

        task_label = "DDK (diadochokinetic)" if task_type == "DDK" else (
            "Vowels" if task_type == "Vowels" else str(task_type)
        )
        lines.append(f"## {task_label}")
        lines.append("")
        include_paper_baseline = task_type == "DDK"
        if include_paper_baseline:
            lines.append(
                "| Lang | Language | Model | Features | Specificity | Sensitivity | F1 | "
                "Paper Spec | Paper Sens | Paper F1 |"
            )
            lines.append(
                "|------|----------|-------|----------|-------------|-------------|-----|"
                "-------------|-------------|----------|"
            )
        else:
            lines.append(
                "| Lang | Language | Model | Features | Specificity | Sensitivity | F1 |"
            )
            lines.append(
                "|------|----------|-------|----------|-------------|-------------|-----|"
            )

        langs = [l for l in LANG_ORDER if l in section["language"].values]
        langs += sorted(set(section["language"]) - set(langs))

        for lang in langs:
            row = section[section["language"] == lang]
            if row.empty:
                continue
            r = row.iloc[0]
            model = str(r.get("model", "")).upper()
            feat = str(r.get("feature_subset", ""))
            spec = _fmt_mean_std(
                r.get("paper_specificity_mean", np.nan),
                r.get("paper_specificity_std", np.nan),
            )
            sens = _fmt_mean_std(
                r.get("paper_sensitivity_mean", np.nan),
                r.get("paper_sensitivity_std", np.nan),
            )
            f1 = _fmt_mean_std(
                r.get("paper_f1_mean", np.nan),
                r.get("paper_f1_std", np.nan),
            )
            lang_name = LANGUAGE_NAMES.get(lang, lang)
            if include_paper_baseline:
                lines.append(
                    f"| {lang} | {lang_name} | {model} | `{feat}` | {spec} | {sens} | {f1} | "
                    f"{_paper_baseline_cell(lang, 'specificity')} | "
                    f"{_paper_baseline_cell(lang, 'sensitivity')} | "
                    f"{_paper_baseline_cell(lang, 'f1')} |"
                )
            else:
                lines.append(
                    f"| {lang} | {lang_name} | {model} | `{feat}` | {spec} | {sens} | {f1} |"
                )

        lines.append("")
        if include_paper_baseline:
            lines.append(
                "*Paper baseline: Hernandez et al., Table 3 monolingual HuBERT-Large, DDK "
                "([arXiv:2603.22225v2](https://arxiv.org/abs/2603.22225)).*"
            )
            lines.append("")

        for lang in langs:
            row = section[section["language"] == lang]
            if row.empty:
                continue
            r = row.iloc[0]
            lang_name = LANGUAGE_NAMES.get(lang, lang)
            lines.append(f"### {lang} — {lang_name}")
            lines.append("")
            lines.append(
                f"- **Best configuration:** {str(r.get('model', '')).upper()} + `{r.get('feature_subset', '')}`"
            )
            lines.append(f"- **Source CSV:** `{r.get('csv_file', '')}`")
            if "n_rows" in r and not pd.isna(r.get("n_rows")):
                lines.append(f"- **Samples:** {int(r['n_rows'])}")
            if "n_features" in r and not pd.isna(r.get("n_features")):
                lines.append(f"- **Features used:** {int(r['n_features'])}")
            lines.append(
                f"- **Specificity:** {_fmt_mean_std(r.get('paper_specificity_mean', np.nan), r.get('paper_specificity_std', np.nan))}"
            )
            lines.append(
                f"- **Sensitivity:** {_fmt_mean_std(r.get('paper_sensitivity_mean', np.nan), r.get('paper_sensitivity_std', np.nan))}"
            )
            lines.append(
                f"- **F1:** {_fmt_mean_std(r.get('paper_f1_mean', np.nan), r.get('paper_f1_std', np.nan))}"
            )
            if "speaker_balanced_acc_mean" in r and not pd.isna(r.get("speaker_balanced_acc_mean")):
                lines.append(
                    f"- **Speaker balanced accuracy (0.5 threshold):** "
                    f"{r['speaker_balanced_acc_mean']:.3f} ± {r.get('speaker_balanced_acc_std', 0):.3f}"
                )
            lines.append("")

    md_path = run_output_dir / "best_results_by_language.md"
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return md_path


def save_and_print_paper_style_reports(
    summary_df: pd.DataFrame,
    run_output_dir: Path,
    min_sens: float,
    feature_subset: str = "glottal_plus_direct",
    models: list = None,
):
    """
    Print and save paper-style tables (arXiv:2603.22225v2 format): by language and metric.
    Requires --paper_min_sens and paper_* columns in summary_df.
    """
    paper_df = _prepare_paper_summary_table(summary_df)
    if paper_df.empty:
        print("\nPaper-style tables skipped: no paper metrics in summary (use --paper_min_sens 0.9).")
        return

    paper_df = paper_df.sort_values(["task_type", "model", "feature_subset", "language"])
    paper_df.to_csv(run_output_dir / "paper_metrics_by_language.csv", index=False)

    long_rows = []
    for _, r in paper_df.iterrows():
        for metric_name, mean_col, std_col in _paper_metric_columns():
            long_rows.append({
                "task_type": r.get("task_type"),
                "language": r.get("language"),
                "model": r.get("model"),
                "feature_subset": r.get("feature_subset"),
                "metric": metric_name,
                "mean": r.get(mean_col, np.nan),
                "std": r.get(std_col, np.nan),
                "formatted": _fmt_mean_std(r.get(mean_col, np.nan), r.get(std_col, np.nan)),
            })
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(run_output_dir / "paper_metrics_long.csv", index=False)

    # Pivot: one CSV per (task_type, model, feature_subset) with languages as rows
    pivot_dir = run_output_dir / "paper_tables"
    pivot_dir.mkdir(parents=True, exist_ok=True)

    models_filter = models if models else sorted(paper_df["model"].unique())
    task_types = sorted(paper_df["task_type"].unique())
    feature_subsets = sorted(paper_df["feature_subset"].unique())

    print("\n" + "=" * 80)
    print("PAPER-STYLE RESULTS (compare to arXiv:2603.22225v2 Tables 2–3)")
    print("Columns: Specificity | Sensitivity | F1  (speaker-level, mean ± std over outer folds)")
    print("=" * 80)

    for task_type in task_types:
        for model in models_filter:
            for feat in feature_subsets:
                mask = (
                    (paper_df["task_type"] == task_type)
                    & (paper_df["model"] == model)
                    & (paper_df["feature_subset"] == feat)
                )
                sub = paper_df.loc[mask]
                if sub.empty:
                    continue
                title = f"{task_type} | {model.upper()} | {feat}"
                print_paper_style_language_table(sub, title, min_sens)

                pivot = sub.set_index("language")[
                    [
                        "paper_specificity_mean",
                        "paper_specificity_std",
                        "paper_sensitivity_mean",
                        "paper_sensitivity_std",
                        "paper_f1_mean",
                        "paper_f1_std",
                    ]
                ].copy()
                pivot.columns = [
                    "spec_mean",
                    "spec_std",
                    "sens_mean",
                    "sens_std",
                    "f1_mean",
                    "f1_std",
                ]
                safe_name = f"{task_type}_{model}_{feat}".replace("/", "_")
                pivot.to_csv(pivot_dir / f"{safe_name}.csv")

    # Highlight default feature subset (matches paper QCP glottal + direct features)
    highlight = paper_df[paper_df["feature_subset"] == feature_subset]
    if not highlight.empty:
        print("\n" + "-" * 80)
        print(f"PRIMARY COMPARISON (feature subset: {feature_subset})")
        print("-" * 80)
        for task_type in sorted(highlight["task_type"].unique()):
            for model in models_filter:
                sub = highlight[(highlight["task_type"] == task_type) & (highlight["model"] == model)]
                if sub.empty:
                    continue
                print_paper_style_language_table(
                    sub,
                    f"{task_type} | {model.upper()} | {feature_subset}",
                    min_sens,
                )

    md_path = write_best_results_markdown(paper_df, run_output_dir, min_sens)

    print(f"\nPaper-style CSVs saved under: {run_output_dir}")
    print(f"  - paper_metrics_by_language.csv")
    print(f"  - paper_metrics_long.csv")
    print(f"  - paper_tables/*.csv")
    print(f"  - best_results_by_language.csv")
    if md_path is not None:
        print(f"  - {md_path.name}")


def save_explanatory_plots(summary_df: pd.DataFrame, out_dir: Path, task_name: str):
    if summary_df.empty:
        return

    metrics_to_plot = ["speaker_auc_mean", "speaker_pr_auc_mean", "speaker_f1_mean", "speaker_balanced_acc_mean"]
    available_metrics = [m for m in metrics_to_plot if m in summary_df.columns]
    if not available_metrics:
        return

    labels = (summary_df["model"] + " | " + summary_df["feature_subset"]).tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(max(10, len(labels) * 0.7), 3.6 * len(available_metrics)))
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        ax.bar(x, summary_df[metric].values)
        ax.set_title(f"{task_name}: {metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--', alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_dir / "summary_barplots.png", dpi=180, bbox_inches='tight')
    plt.close(fig)

    # Heatmap-style overview (matshow) for speaker metrics
    heat_cols = [c for c in ["speaker_accuracy_mean", "speaker_balanced_acc_mean", "speaker_f1_mean", "speaker_auc_mean", "speaker_pr_auc_mean"] if c in summary_df.columns]
    if heat_cols:
        mat = summary_df[heat_cols].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(1.8 * len(heat_cols) + 4, 0.5 * len(summary_df) + 3))
        im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
        ax.set_title(f"{task_name}: Speaker-level metric overview")
        ax.set_xticks(np.arange(len(heat_cols)))
        ax.set_xticklabels(heat_cols, rotation=30, ha='right')
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('score')
        fig.tight_layout()
        fig.savefig(out_dir / "summary_heatmap.png", dpi=180, bbox_inches='tight')
        plt.close(fig)


def evaluate_model_with_nested_cv(
    model_name,
    estimator,
    param_grid,
    X,
    y,
    groups,
    n_repeats=1,
    n_outer_splits=10,
    n_inner_splits=5,
    base_random_state=42,
    outer_cv_mode: str = "sgkf",
):
    metrics_template = {
        "accuracy": [],
        "balanced_acc": [],
        "f1": [],
        "auc": [],
        "pr_auc": [],
        "precision": [],
        "recall": [],
    }

    results_sample = {k: [] for k in metrics_template}
    results_speaker = {k: [] for k in metrics_template}
    fold_records = []

    # Optional "paper-style" speaker metrics (threshold tuned to hit min sensitivity).
    paper_min_sens = globals().get("PAPER_MIN_SENS", None)
    results_speaker_paper = None
    if paper_min_sens is not None:
        results_speaker_paper = {
            "f1": [],
            "sensitivity": [],
            "specificity": [],
        }

    for repeat in range(n_repeats):
        current_seed = base_random_state + repeat

        outer_mode = str(outer_cv_mode or "sgkf").strip().lower()
        if outer_mode in {"loso", "leaveonegroupout", "logo"}:
            outer_cv = LeaveOneGroupOut()
            outer_split_iter = outer_cv.split(X, y, groups)
            outer_total = int(len(np.unique(groups)))
        else:
            outer_cv = StratifiedGroupKFold(n_splits=n_outer_splits, shuffle=True, random_state=current_seed)
            outer_split_iter = outer_cv.split(X, y, groups)
            outer_total = int(n_outer_splits)
        inner_cv = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True, random_state=current_seed)

        for fold, (train_idx, test_idx) in enumerate(outer_split_iter, start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]

            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=inner_cv.split(X_train, y_train, g_train),
                scoring="roc_auc",
                n_jobs=int(os.environ.get("CLASSIFY_N_JOBS", "-1")),
                verbose=0,
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            y_pred_sample = best_model.predict(X_test)
            y_proba_sample = best_model.predict_proba(X_test)[:, 1]

            results_sample["accuracy"].append(accuracy_score(y_test, y_pred_sample))
            results_sample["balanced_acc"].append(balanced_accuracy_score(y_test, y_pred_sample))
            results_sample["f1"].append(f1_score(y_test, y_pred_sample, zero_division=0))
            results_sample["precision"].append(precision_score(y_test, y_pred_sample, zero_division=0))
            results_sample["recall"].append(recall_score(y_test, y_pred_sample, zero_division=0))

            if len(np.unique(y_test)) > 1:
                results_sample["auc"].append(roc_auc_score(y_test, y_proba_sample))
                results_sample["pr_auc"].append(average_precision_score(y_test, y_proba_sample))
            else:
                results_sample["auc"].append(np.nan)
                results_sample["pr_auc"].append(np.nan)

            y_test_spk, y_proba_spk, _ = aggregate_mean_by_group(y_test, y_proba_sample, g_test)
            y_pred_spk = (y_proba_spk >= 0.5).astype(int)

            results_speaker["accuracy"].append(accuracy_score(y_test_spk, y_pred_spk))
            results_speaker["balanced_acc"].append(balanced_accuracy_score(y_test_spk, y_pred_spk))
            results_speaker["f1"].append(f1_score(y_test_spk, y_pred_spk, zero_division=0))
            results_speaker["precision"].append(precision_score(y_test_spk, y_pred_spk, zero_division=0))
            results_speaker["recall"].append(recall_score(y_test_spk, y_pred_spk, zero_division=0))

            if len(np.unique(y_test_spk)) > 1:
                results_speaker["auc"].append(roc_auc_score(y_test_spk, y_proba_spk))
                results_speaker["pr_auc"].append(average_precision_score(y_test_spk, y_proba_spk))
            else:
                results_speaker["auc"].append(np.nan)
                results_speaker["pr_auc"].append(np.nan)

            paper_thr = np.nan
            if results_speaker_paper is not None:
                # Inner-CV threshold selection on speaker-aggregated validation predictions.
                inner_y_all = []
                inner_p_all = []
                inner_g_all = []
                for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train, g_train):
                    X_tr, X_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
                    y_tr, y_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]
                    g_val = g_train.iloc[inner_val_idx]

                    m = grid_search.best_estimator_
                    m.fit(X_tr, y_tr)
                    p_val = m.predict_proba(X_val)[:, 1]
                    inner_y_all.append(y_val.to_numpy())
                    inner_p_all.append(p_val)
                    inner_g_all.append(g_val.to_numpy())

                inner_y = np.concatenate(inner_y_all) if inner_y_all else np.array([], dtype=int)
                inner_p = np.concatenate(inner_p_all) if inner_p_all else np.array([], dtype=float)
                inner_g = np.concatenate(inner_g_all) if inner_g_all else np.array([], dtype=object)

                if inner_y.size > 0:
                    inner_y_spk, inner_p_spk, _ = aggregate_mean_by_group(inner_y, inner_p, inner_g)
                    paper_thr = _select_threshold_max_spec_at_min_sens(inner_y_spk, inner_p_spk, float(paper_min_sens))
                else:
                    paper_thr = 0.5

                y_pred_spk_paper = (y_proba_spk >= paper_thr).astype(int)
                sens, spec = _sens_spec_from_labels(y_test_spk, y_pred_spk_paper)
                results_speaker_paper["sensitivity"].append(sens)
                results_speaker_paper["specificity"].append(spec)
                results_speaker_paper["f1"].append(f1_score(y_test_spk, y_pred_spk_paper, zero_division=0))

            fold_records.append({
                "model_name": model_name,
                "repeat": repeat + 1,
                "fold": fold,
                "sample_accuracy": results_sample["accuracy"][-1],
                "sample_balanced_acc": results_sample["balanced_acc"][-1],
                "sample_f1": results_sample["f1"][-1],
                "sample_auc": results_sample["auc"][-1],
                "sample_pr_auc": results_sample["pr_auc"][-1],
                "sample_precision": results_sample["precision"][-1],
                "sample_recall": results_sample["recall"][-1],
                "speaker_accuracy": results_speaker["accuracy"][-1],
                "speaker_balanced_acc": results_speaker["balanced_acc"][-1],
                "speaker_f1": results_speaker["f1"][-1],
                "speaker_auc": results_speaker["auc"][-1],
                "speaker_pr_auc": results_speaker["pr_auc"][-1],
                "speaker_precision": results_speaker["precision"][-1],
                "speaker_recall": results_speaker["recall"][-1],
                "paper_min_sens": paper_min_sens if results_speaker_paper is not None else np.nan,
                "paper_threshold": paper_thr,
                "paper_speaker_sensitivity": results_speaker_paper["sensitivity"][-1] if results_speaker_paper is not None else np.nan,
                "paper_speaker_specificity": results_speaker_paper["specificity"][-1] if results_speaker_paper is not None else np.nan,
                "paper_speaker_f1": results_speaker_paper["f1"][-1] if results_speaker_paper is not None else np.nan,
            })

            print(f"{model_name} | Fold {fold}/{outer_total} complete")

    if results_speaker_paper is not None:
        print(f"\nPaper-style speaker metrics (threshold tuned for sensitivity ≥ {paper_min_sens})")
        for k, vals in results_speaker_paper.items():
            mean_v = float(np.nanmean(vals)) if len(vals) else float("nan")
            std_v = float(np.nanstd(vals)) if len(vals) else float("nan")
            print(f"{k:<12}: {mean_v:.4f} ± {std_v:.4f}")

    return results_sample, results_speaker, pd.DataFrame(fold_records)


def _read_csv_if_nonempty(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Read a CSV path, returning None if the file is missing or has no rows."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty or len(df.columns) == 0:
        return None
    return df


def _concat_nonempty_csvs(paths: List[Path]) -> pd.DataFrame:
    frames = [_read_csv_if_nonempty(p) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_feature_subsets(X: pd.DataFrame):
    glottal_base_roots = ['NAQ', 'QOQ', 'HRF', 'H1H2']
    direct_roots = ['G_RMS', 'G_ZCR', 'G_CREST', 'DG_PEAK', 'RES_RMS']

    glottal_base_cols = []
    direct_cols = []
    mfcc_cols = []
    hubert_cols = []

    for col in X.columns:
        if col.startswith('hubert_'):
            hubert_cols.append(col)
            continue
        if col.startswith('mfcc_'):
            mfcc_cols.append(col)
            continue

        if any(col.startswith(f"{r}_") for r in glottal_base_roots):
            glottal_base_cols.append(col)
        elif any(col.startswith(f"{r}_") for r in direct_roots):
            direct_cols.append(col)

    glottal_plus_direct_cols = glottal_base_cols + direct_cols
    subsets = {
        'glottal_only': glottal_base_cols,
        'glottal_plus_direct': glottal_plus_direct_cols,
        'glottal_plus_direct_plus_mfcc': glottal_plus_direct_cols + mfcc_cols,
        'glottal_plus_mfcc': glottal_base_cols + mfcc_cols,
        'hubert_all': hubert_cols,
        'hubert_plus_glottal_plus_direct': hubert_cols + glottal_plus_direct_cols,
    }

    # Preserve original order from X.columns for reproducibility
    ordered_subsets = {}
    for name, cols in subsets.items():
        col_set = set(cols)
        ordered_subsets[name] = [c for c in X.columns if c in col_set]

    return ordered_subsets

def run_classification_for_task(csv_file, run_output_dir: Path):
    print(f"\n{'='*80}")
    print(f"Processing task from file: {csv_file}")
    print(f"{'='*80}")
    
    data = pd.read_csv(csv_file)

    # Robust preprocessing for engineered feature tables
    required_cols = ['speaker', 'label']
    for col in required_cols:
        if col not in data.columns:
            print(f"Missing required column '{col}' in {csv_file}. Skipping.")
            return

    # Normalize metadata text columns
    for col in ['file_name', 'speaker', 'label', 'task']:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()

    # Optional task filtering (e.g., DDK-only: --task_suffix _ddk)
    task_suffix = str(globals().get("TASK_SUFFIX", "") or "").strip()
    task_contains = str(globals().get("TASK_CONTAINS", "") or "").strip()
    if "task" in data.columns and (task_suffix or task_contains):
        before = len(data)
        mask = pd.Series(True, index=data.index)
        if task_suffix:
            mask &= data["task"].astype(str).str.lower().str.endswith(task_suffix.lower(), na=False)
        if task_contains:
            mask &= data["task"].astype(str).str.lower().str.contains(task_contains.lower(), na=False)
        filtered = data[mask].copy()
        after = len(filtered)
        if after == 0 and is_dedicated_vowels_csv(csv_file):
            print(
                f"Task filter (suffix={task_suffix!r}, contains={task_contains!r}) matched 0 rows; "
                f"{Path(csv_file).name} is vowels-only — keeping all {before} rows."
            )
        elif after == 0:
            print(
                f"Task filter applied (suffix={task_suffix!r}, contains={task_contains!r}): "
                f"0/{before} rows kept. Skipping."
            )
            return
        else:
            data = filtered
            kept_tasks = sorted(data["task"].astype(str).unique().tolist())
            print(
                f"Task filter applied (suffix={task_suffix!r}, contains={task_contains!r}): "
                f"{after}/{before} rows kept | tasks={kept_tasks}"
            )

    # Recover missing metadata for Vowels-style IDs (e.g., AVPEPUDEAC0001a1 / AVPEPUDEA0001a1)
    if 'file_name' in data.columns:
        file_ids = data['file_name'].astype(str)
        unknown_label = data['label'].astype(str).str.upper().eq('UNKNOWN') | data['label'].isna()
        inferred_label = pd.Series(index=data.index, dtype='object')
        inferred_label[file_ids.str.match(r'^AVPEPUDEAC\d{4}[aeiou]\d$', na=False)] = 'C'
        inferred_label[file_ids.str.match(r'^AVPEPUDEA\d{4}[aeiou]\d$', na=False)] = inferred_label[file_ids.str.match(r'^AVPEPUDEA\d{4}[aeiou]\d$', na=False)].fillna('A')
        data.loc[unknown_label, 'label'] = inferred_label[unknown_label].fillna(data.loc[unknown_label, 'label'])

        unknown_speaker = data['speaker'].astype(str).str.upper().eq('UNKNOWN') | data['speaker'].isna()
        c_digits = file_ids.str.extract(r'^AVPEPUDEAC(\d{4})', expand=False)
        a_digits = file_ids.str.extract(r'^AVPEPUDEA(\d{4})', expand=False)
        inferred_speaker = pd.Series(index=data.index, dtype='object')
        inferred_speaker[c_digits.notna()] = 'C' + c_digits[c_digits.notna()]
        inferred_speaker[a_digits.notna()] = inferred_speaker[a_digits.notna()].fillna('A' + a_digits[a_digits.notna()])
        data.loc[unknown_speaker, 'speaker'] = inferred_speaker[unknown_speaker].fillna(data.loc[unknown_speaker, 'speaker'])

    # Labels
    data['label_num'] = data['label'].map({'A': 1, 'C': 0})

    # Keep only rows with valid labels and speaker IDs
    initial_len = len(data)
    data = data.dropna(subset=['label_num', 'speaker'])
    dropped_invalid_meta = initial_len - len(data)

    # Build feature matrix
    cols_to_drop = ['file_name', 'speaker', 'label', 'task', 'label_num']
    feature_cols = [c for c in data.columns if c not in cols_to_drop]
    X = data[feature_cols].apply(pd.to_numeric, errors='coerce')

    # Drop feature columns that are entirely NaN (common for some high-order stats)
    n_cols_before = X.shape[1]
    X = X.dropna(axis=1, how='all')
    dropped_all_nan_cols = n_cols_before - X.shape[1]

    # Impute remaining NaNs with column medians (fold-safe scaling still happens in pipeline)
    nan_before_impute = int(X.isna().sum().sum())
    if nan_before_impute > 0:
        X = X.fillna(X.median(numeric_only=True))
    nan_after_impute = int(X.isna().sum().sum())

    print(
        f"Rows kept: {len(data)} (dropped invalid label/speaker: {dropped_invalid_meta}) | "
        f"Feature cols: {X.shape[1]} (dropped all-NaN cols: {dropped_all_nan_cols}) | "
        f"NaNs imputed: {nan_before_impute - nan_after_impute}"
    )

    if len(data) < 20 or X.shape[1] == 0:
        print("Not enough usable data/features to perform 10-fold CV. Skipping.")
        return

    y = data['label_num']
    groups = data['speaker']
    
    # CONFIGURATION
    # Allow CLI override via module-level globals (set in __main__).
    N_REPEATS = int(globals().get("N_REPEATS", 1))
    N_OUTER_SPLITS = int(globals().get("N_OUTER_SPLITS", 10))
    N_INNER_SPLITS = int(globals().get("N_INNER_SPLITS", 5))
    BASE_RANDOM_STATE = 42

    feature_subsets = build_feature_subsets(X)

    models_config = {}

    if 'svm' in RUN_MODELS:
        svm_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, random_state=BASE_RANDOM_STATE, class_weight='balanced')),
        ])
        svm_grid = {
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ['scale', 1, 0.1, 0.01, 0.001],
            "svc__kernel": ['rbf'],
        }
        models_config['svm'] = {
            'title': 'SVM (RBF)',
            'model_name': 'SVM',
            'estimator': svm_pipe,
            'param_grid': svm_grid,
        }

    if 'rf' in RUN_MODELS:
        rf = RandomForestClassifier(random_state=BASE_RANDOM_STATE, class_weight='balanced')
        rf_grid = {
            "n_estimators": [200, 400],
            "max_depth": [None, 8, 16],
            "min_samples_leaf": [1, 3, 5],
        }
        models_config['rf'] = {
            'title': 'Random Forest',
            'model_name': 'RF',
            'estimator': rf,
            'param_grid': rf_grid,
        }

    if 'xgb' in RUN_MODELS:
        xgb = XGBClassifier(
            random_state=BASE_RANDOM_STATE,
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1,
        )
        xgb_grid = {
            'n_estimators': [200, 400],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.03, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
        models_config['xgb'] = {
            'title': 'XGBoost',
            'model_name': 'XGB',
            'estimator': xgb,
            'param_grid': xgb_grid,
        }

    if len(models_config) == 0:
        print("No valid models selected in RUN_MODELS. Use one or more of: ['svm', 'rf', 'xgb']")
        return

    print(f"\nFINAL RESULTS FOR {csv_file}")

    task_stem = Path(csv_file).stem
    task_out_dir = run_output_dir / task_stem
    task_out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_fold_dfs = []

    for subset_name, subset_cols in feature_subsets.items():
        if len(subset_cols) == 0:
            print(f"\nSubset {subset_name}: no columns found, skipping")
            continue

        X_subset = X[subset_cols]
        print(f"\n--- Feature subset: {subset_name} ({len(subset_cols)} features) ---")

        for model_key in RUN_MODELS:
            if model_key not in models_config:
                continue

            model_cfg = models_config[model_key]
            model_sample, model_speaker, fold_df = evaluate_model_with_nested_cv(
                model_name=f"{model_cfg['model_name']}-{subset_name}",
                estimator=model_cfg['estimator'],
                param_grid=model_cfg['param_grid'],
                X=X_subset,
                y=y,
                groups=groups,
                n_repeats=N_REPEATS,
                n_outer_splits=N_OUTER_SPLITS,
                n_inner_splits=N_INNER_SPLITS,
                base_random_state=BASE_RANDOM_STATE,
                outer_cv_mode=str(globals().get("OUTER_CV_MODE", "sgkf")),
            )

            summarize_metrics(f"{model_cfg['title']} [{subset_name}]", model_sample, model_speaker)

            row = {
                "csv_file": csv_file,
                "task_stem": task_stem,
                "language": infer_language_code(csv_file),
                "task_type": infer_task_type(csv_file),
                "model": model_key,
                "model_title": model_cfg['title'],
                "feature_subset": subset_name,
                "n_features": len(subset_cols),
                "n_rows": len(data),
            }
            row.update(compute_metrics_summary_row(model_sample, model_speaker))
            row.update(compute_paper_metrics_summary_row(fold_df))
            summary_rows.append(row)

            if not fold_df.empty:
                fold_df["csv_file"] = csv_file
                fold_df["task_stem"] = task_stem
                fold_df["model"] = model_key
                fold_df["feature_subset"] = subset_name
                all_fold_dfs.append(fold_df)

    if not summary_rows:
        print(
            "WARNING: No feature subsets produced results. "
            "For HuBERT tables ensure columns are named hubert_0, hubert_1, ..."
        )
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = task_out_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)

    if all_fold_dfs:
        folds_df = pd.concat(all_fold_dfs, ignore_index=True)
        folds_df.to_csv(task_out_dir / "fold_metrics.csv", index=False)

    save_explanatory_plots(summary_df, task_out_dir, task_stem)
    print(f"Saved task outputs to: {task_out_dir}")

if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(description="Run classification on feature CSVs")
    _parser.add_argument("--csv", nargs="+", default=None, help="Specific CSV files to process (default: all features_*.csv)")
    _parser.add_argument(
        "--models",
        type=str,
        default=",".join(RUN_MODELS),
        help="Comma-separated models to run: svm,rf,xgb (default matches script default).",
    )
    _parser.add_argument(
        "--outer_cv",
        type=str,
        default="sgkf",
        help="Outer CV mode: 'sgkf' (StratifiedGroupKFold) or 'loso' (Leave-One-Speaker-Out).",
    )
    _parser.add_argument("--outer_splits", type=int, default=10, help="Outer StratifiedGroupKFold splits (default: 10).")
    _parser.add_argument("--inner_splits", type=int, default=5, help="Inner StratifiedGroupKFold splits for GridSearch (default: 5).")
    _parser.add_argument("--repeats", type=int, default=1, help="Repeat nested CV with different seeds (default: 1).")
    _parser.add_argument(
        "--paper_min_sens",
        type=float,
        default=None,
        help="If set (e.g., 0.9), also report paper-style speaker metrics using a threshold tuned in inner CV to satisfy sensitivity >= this value (maximize specificity).",
    )
    _parser.add_argument(
        "--task_suffix",
        type=str,
        default="",
        help="Optional: keep only rows whose 'task' ends with this suffix (e.g. '_ddk').",
    )
    _parser.add_argument(
        "--task_contains",
        type=str,
        default="",
        help="Optional: keep only rows whose 'task' contains this substring (case-insensitive).",
    )
    _parser.add_argument(
        "--paper_feature_subset",
        type=str,
        default="glottal_plus_direct",
        help="Feature subset highlighted in the primary paper-style comparison table (default: glottal_plus_direct).",
    )
    _parser.add_argument(
        "--report_from_run",
        type=str,
        default=None,
        help="Regenerate paper-style tables from an existing run directory (no re-training).",
    )
    _args = _parser.parse_args()

    if _args.report_from_run:
        run_output_dir = Path(_args.report_from_run)
        if not run_output_dir.is_dir():
            raise SystemExit(f"Run directory not found: {run_output_dir}")

        min_sens = float(_args.paper_min_sens) if _args.paper_min_sens is not None else 0.9
        summary_path = run_output_dir / "global_summary_metrics.csv"
        fold_files = sorted(run_output_dir.glob("*/fold_metrics.csv"))

        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            if "paper_f1_mean" not in summary_df.columns and fold_files:
                paper_from_folds = build_paper_summary_from_folds(
                    _concat_nonempty_csvs(fold_files)
                )
                if not paper_from_folds.empty:
                    merge_keys = ["csv_file", "model", "feature_subset"]
                    summary_df = summary_df.drop(
                        columns=[c for c in summary_df.columns if c.startswith("paper_")],
                        errors="ignore",
                    )
                    summary_df = summary_df.merge(paper_from_folds, on=merge_keys, how="left", suffixes=("", "_fold"))
        elif fold_files:
            summary_df = build_paper_summary_from_folds(
                _concat_nonempty_csvs(fold_files)
            )
        else:
            raise SystemExit(f"No summary or fold_metrics found in {run_output_dir}")

        _models = [m.strip().lower() for m in str(_args.models).split(",") if m.strip()]
        save_and_print_paper_style_reports(
            summary_df,
            run_output_dir,
            min_sens=min_sens,
            feature_subset=str(_args.paper_feature_subset),
            models=_models or None,
        )
        raise SystemExit(0)

    # Override global run config from CLI
    _models = [m.strip().lower() for m in str(_args.models).split(",") if m.strip()]
    if _models:
        RUN_MODELS[:] = _models

    globals()["OUTER_CV_MODE"] = str(_args.outer_cv).strip().lower()
    globals()["TASK_SUFFIX"] = str(_args.task_suffix or "").strip()
    globals()["TASK_CONTAINS"] = str(_args.task_contains or "").strip()
    globals()["PAPER_MIN_SENS"] = _args.paper_min_sens

    base_results_dir = Path("classification_results")
    base_results_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = base_results_dir / f"run_{run_timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing results under: {run_output_dir}")

    csv_files = _args.csv if _args.csv else glob.glob("features_*.csv")
    if not csv_files:
        print("No feature CSV files found. Please run the MATLAB extraction script first.")
    else:
        for csv_file in csv_files:
            # Patch nested-CV configuration for this run
            # (kept as globals inside run_classification_for_task for minimal diff)
            global_N_OUTER_SPLITS = int(_args.outer_splits)
            global_N_INNER_SPLITS = int(_args.inner_splits)
            global_N_REPEATS = int(_args.repeats)

            # Monkey-patch via module globals used in run_classification_for_task
            # (avoid refactoring the full file; keeps existing behavior as default)
            globals()["N_OUTER_SPLITS"] = global_N_OUTER_SPLITS
            globals()["N_INNER_SPLITS"] = global_N_INNER_SPLITS
            globals()["N_REPEATS"] = global_N_REPEATS

            run_classification_for_task(csv_file, run_output_dir)

        # Run-level global exports for easy cross-task comparison
        summary_files = sorted(run_output_dir.glob("*/summary_metrics.csv"))
        all_summary = _concat_nonempty_csvs(summary_files)
        if not all_summary.empty:
            all_summary.to_csv(run_output_dir / "global_summary_metrics.csv", index=False)

            # Convenience ranking by speaker AUC (descending)
            if "speaker_auc_mean" in all_summary.columns:
                ranked = all_summary.sort_values(by="speaker_auc_mean", ascending=False)
                ranked.to_csv(run_output_dir / "global_summary_ranked_by_speaker_auc.csv", index=False)

        fold_files = sorted(run_output_dir.glob("*/fold_metrics.csv"))
        all_folds = _concat_nonempty_csvs(fold_files)
        if not all_folds.empty:
            all_folds.to_csv(run_output_dir / "global_fold_metrics.csv", index=False)

        if _args.paper_min_sens is not None and not all_summary.empty:
            save_and_print_paper_style_reports(
                all_summary,
                run_output_dir,
                min_sens=float(_args.paper_min_sens),
                feature_subset=str(_args.paper_feature_subset),
                models=_models,
            )

        print(f"Global summary files saved in: {run_output_dir}")
