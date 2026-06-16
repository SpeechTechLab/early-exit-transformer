#!/usr/bin/env python3
"""Late fusion for DDK classification: weighted sum of modality posteriors (Σw = 1).

Trains separate classifiers on HuBERT, Whisper, and glottal (QCP) features with the
same nested speaker-grouped CV as ``classify_tasks.py``. Fusion weights are tuned on
inner-CV speaker posteriors; test-fold metrics follow the paper protocol (sens ≥ 0.90).

Example:
  python3 late_fusion_ddk.py --presets LF-H+W+G,LF-H+W-CTC+G --models rf,xgb
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from xgboost import XGBClassifier

from classify_tasks import (
    aggregate_mean_by_group,
    compute_metrics_summary_row,
    compute_paper_metrics_summary_row,
    infer_language_code,
    _select_threshold_max_spec_at_min_sens,
    _sens_spec_from_labels,
)

_REPO = Path(__file__).resolve().parent
META_COLS = ["file_name", "speaker", "label", "task"]
LANGS = ("German", "Czech", "Colombian")


@dataclass(frozen=True)
class FusionPreset:
    name: str
    whisper_csv_suffix: str  # e.g. whisper_DDK, whisper_ft_DDK, whisper_ctc_DDK


FUSION_PRESETS: Dict[str, FusionPreset] = {
    "LF-H+W+G": FusionPreset("LF-H+W+G", "whisper_DDK"),
    "LF-H+W-FT+G": FusionPreset("LF-H+W-FT+G", "whisper_ft_DDK"),
    "LF-H+W-CTC+G": FusionPreset("LF-H+W-CTC+G", "whisper_ctc_DDK"),
}


def _weight_grid(n_modalities: int, step: float = 0.1) -> List[np.ndarray]:
    if n_modalities == 2:
        w = np.arange(0.0, 1.0 + 1e-9, step)
        return [np.array([a, 1.0 - a], dtype=float) for a in w]
    if n_modalities == 3:
        out: List[np.ndarray] = []
        for w0 in np.arange(0.0, 1.0 + 1e-9, step):
            for w1 in np.arange(0.0, 1.0 - w0 + 1e-9, step):
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9:
                    continue
                out.append(np.array([w0, w1, max(0.0, w2)], dtype=float))
        return out
    raise ValueError(f"Unsupported n_modalities={n_modalities}")


def _tune_weights_speaker_auc(
    y_spk: np.ndarray,
    prob_matrix: np.ndarray,
    *,
    step: float = 0.1,
) -> np.ndarray:
    """Return weights (sum=1) maximizing speaker-level ROC-AUC on inner validation."""
    y_spk = np.asarray(y_spk).astype(int)
    if len(np.unique(y_spk)) < 2:
        return np.ones(prob_matrix.shape[1], dtype=float) / prob_matrix.shape[1]

    best_w = np.ones(prob_matrix.shape[1], dtype=float) / prob_matrix.shape[1]
    best_auc = -1.0
    for w in _weight_grid(prob_matrix.shape[1], step=step):
        p = np.clip(prob_matrix @ w, 0.0, 1.0)
        auc = roc_auc_score(y_spk, p)
        if auc > best_auc:
            best_auc = auc
            best_w = w.copy()
    return best_w


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for col in META_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def _feature_matrix(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns")
    X = X.fillna(X.median(numeric_only=True))
    return X


def merge_modality_tables(
    hubert_csv: Path,
    whisper_csv: Path,
    glottal_csv: Path,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    h = _load_csv(hubert_csv)
    w = _load_csv(whisper_csv)
    g = _load_csv(glottal_csv)

    hubert_cols = [c for c in h.columns if c.startswith("hubert_")]
    whisper_cols = [c for c in w.columns if c.startswith("whisper_")]
    glottal_cols = [c for c in g.columns if c not in META_COLS]

    if not hubert_cols or not whisper_cols or not glottal_cols:
        raise ValueError(
            f"Missing modality columns in {hubert_csv.name}, {whisper_csv.name}, or {glottal_csv.name}"
        )

    merged = h[META_COLS + hubert_cols].merge(
        w[META_COLS + whisper_cols], on=META_COLS, how="inner", suffixes=("", "_w")
    )
    merged = merged.merge(g[META_COLS + glottal_cols], on=META_COLS, how="inner")
    if merged.empty:
        raise RuntimeError(f"No overlapping rows: {hubert_csv.name}")

    modality_cols = {
        "hubert": hubert_cols,
        "whisper": whisper_cols,
        "glottal": glottal_cols,
    }
    return merged, modality_cols


def _build_estimator(model_key: str, random_state: int = 42):
    if model_key == "rf":
        return (
            RandomForestClassifier(random_state=random_state, class_weight="balanced"),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 3, 5],
            },
            int(os.environ.get("CLASSIFY_N_JOBS", "-1")),
        )
    if model_key == "xgb":
        xgb_device = str(os.environ.get("XGB_DEVICE", "cpu") or "cpu").strip().lower()
        kwargs = {
            "random_state": random_state,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": 1,
        }
        grid_n_jobs = int(os.environ.get("CLASSIFY_N_JOBS", "-1"))
        if xgb_device in {"cuda", "gpu"}:
            kwargs["device"] = "cuda"
            kwargs["tree_method"] = "hist"
            grid_n_jobs = 1
        return (
            XGBClassifier(**kwargs),
            {
                "n_estimators": [200, 400],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            grid_n_jobs,
        )
    raise ValueError(f"Unknown model: {model_key}")


def _clone_with_best_params(model_key: str, best_params: dict, random_state: int = 42):
    est, _, _ = _build_estimator(model_key, random_state)
    est.set_params(**best_params)
    return est


def evaluate_late_fusion(
    data: pd.DataFrame,
    modality_cols: Dict[str, List[str]],
    *,
    model_key: str,
    fusion_name: str,
    csv_tag: str,
    paper_min_sens: float,
    n_outer_splits: int = 10,
    n_inner_splits: int = 5,
    weight_step: float = 0.1,
    random_state: int = 42,
) -> Tuple[dict, dict, pd.DataFrame]:
    data = data.copy()
    data["label_num"] = data["label"].map({"A": 1, "C": 0})
    data = data.dropna(subset=["label_num", "speaker"])
    y = data["label_num"]
    groups = data["speaker"]

    modality_names = ["hubert", "whisper", "glottal"]
    X_parts = {m: _feature_matrix(data, modality_cols[m]) for m in modality_names}

    estimator, param_grid, grid_n_jobs = _build_estimator(model_key, random_state)

    outer_cv = StratifiedGroupKFold(
        n_splits=n_outer_splits, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedGroupKFold(
        n_splits=n_inner_splits, shuffle=True, random_state=random_state
    )

    results_sample = {k: [] for k in ["accuracy", "balanced_acc", "f1", "auc", "pr_auc", "precision", "recall"]}
    results_speaker = {k: [] for k in results_sample}
    results_paper = {"f1": [], "sensitivity": [], "specificity": []}
    fold_records = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_parts["hubert"], y, groups), start=1):
        print(f"{fusion_name} | {model_key} | Fold {fold}/{n_outer_splits} ...", flush=True)

        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]

        # Per-modality grid search on outer-train
        fitted_models = {}
        best_params: Dict[str, dict] = {}
        for m in modality_names:
            X_tr = X_parts[m].iloc[train_idx]
            gs = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=inner_cv.split(X_tr, y_train, g_train),
                scoring="roc_auc",
                n_jobs=grid_n_jobs,
                verbose=0,
            )
            gs.fit(X_tr, y_train)
            fitted_models[m] = gs.best_estimator_
            best_params[m] = dict(gs.best_params_)

        # Tune fusion weights on inner-CV speaker posteriors (reuse best hyperparams)
        inner_y_spk_all: List[np.ndarray] = []
        inner_p_spk_all: List[np.ndarray] = []

        X_train_hubert = X_parts["hubert"].iloc[train_idx]
        for inner_tr, inner_val in inner_cv.split(X_train_hubert, y_train, g_train):
            p_mods = []
            for m in modality_names:
                est = _clone_with_best_params(model_key, best_params[m], random_state)
                X_itr = X_parts[m].iloc[train_idx].iloc[inner_tr]
                y_itr = y_train.iloc[inner_tr]
                est.fit(X_itr, y_itr)
                X_val = X_parts[m].iloc[train_idx].iloc[inner_val]
                p_mods.append(est.predict_proba(X_val)[:, 1])

            y_val = y_train.iloc[inner_val].to_numpy()
            g_val = g_train.iloc[inner_val].to_numpy()
            prob_mat = np.column_stack(
                [
                    aggregate_mean_by_group(y_val, p_mods[j], g_val)[1]
                    for j in range(len(modality_names))
                ]
            )
            y_spk = aggregate_mean_by_group(y_val, p_mods[0], g_val)[0]
            inner_y_spk_all.append(y_spk)
            inner_p_spk_all.append(prob_mat)

        if inner_y_spk_all:
            y_tune = np.concatenate(inner_y_spk_all)
            p_tune = np.vstack(inner_p_spk_all)
            weights = _tune_weights_speaker_auc(y_tune, p_tune, step=weight_step)
        else:
            weights = np.ones(3) / 3.0

        # Test predictions
        p_test_mods = []
        for m in modality_names:
            p_test_mods.append(fitted_models[m].predict_proba(X_parts[m].iloc[test_idx])[:, 1])
        p_test = np.clip(np.column_stack(p_test_mods) @ weights, 0.0, 1.0)
        y_pred_sample = (p_test >= 0.5).astype(int)

        results_sample["accuracy"].append(accuracy_score(y_test, y_pred_sample))
        results_sample["balanced_acc"].append(balanced_accuracy_score(y_test, y_pred_sample))
        results_sample["f1"].append(f1_score(y_test, y_pred_sample, zero_division=0))
        results_sample["precision"].append(precision_score(y_test, y_pred_sample, zero_division=0))
        results_sample["recall"].append(recall_score(y_test, y_pred_sample, zero_division=0))
        if len(np.unique(y_test)) > 1:
            results_sample["auc"].append(roc_auc_score(y_test, p_test))
            results_sample["pr_auc"].append(average_precision_score(y_test, p_test))
        else:
            results_sample["auc"].append(np.nan)
            results_sample["pr_auc"].append(np.nan)

        y_spk, p_spk, _ = aggregate_mean_by_group(y_test.to_numpy(), p_test, g_test.to_numpy())
        y_pred_spk = (p_spk >= 0.5).astype(int)
        results_speaker["accuracy"].append(accuracy_score(y_spk, y_pred_spk))
        results_speaker["balanced_acc"].append(balanced_accuracy_score(y_spk, y_pred_spk))
        results_speaker["f1"].append(f1_score(y_spk, y_pred_spk, zero_division=0))
        results_speaker["precision"].append(precision_score(y_spk, y_pred_spk, zero_division=0))
        results_speaker["recall"].append(recall_score(y_spk, y_pred_spk, zero_division=0))
        if len(np.unique(y_spk)) > 1:
            results_speaker["auc"].append(roc_auc_score(y_spk, p_spk))
            results_speaker["pr_auc"].append(average_precision_score(y_spk, p_spk))
        else:
            results_speaker["auc"].append(np.nan)
            results_speaker["pr_auc"].append(np.nan)

        paper_thr = _select_threshold_max_spec_at_min_sens(y_spk, p_spk, paper_min_sens)
        y_pred_paper = (p_spk >= paper_thr).astype(int)
        sens, spec = _sens_spec_from_labels(y_spk, y_pred_paper)
        results_paper["sensitivity"].append(sens)
        results_paper["specificity"].append(spec)
        results_paper["f1"].append(f1_score(y_spk, y_pred_paper, zero_division=0))

        fold_records.append(
            {
                "fold": fold,
                "fusion": fusion_name,
                "model": model_key,
                "w_hubert": float(weights[0]),
                "w_whisper": float(weights[1]),
                "w_glottal": float(weights[2]),
                "sample_f1": results_sample["f1"][-1],
                "speaker_f1": results_speaker["f1"][-1],
                "paper_speaker_f1": results_paper["f1"][-1],
                "paper_speaker_sensitivity": sens,
                "paper_speaker_specificity": spec,
                "paper_threshold": paper_thr,
                "paper_min_sens": paper_min_sens,
            }
        )

    fold_df = pd.DataFrame(fold_records)
    fold_df["csv_file"] = csv_tag
    return results_sample, results_speaker, fold_df


def _paths_for_lang(lang: str, preset: FusionPreset) -> Tuple[Path, Path, Path]:
    hubert = _REPO / f"features_{lang}_hubert_DDK.csv"
    whisper = _REPO / f"features_{lang}_{preset.whisper_csv_suffix}.csv"
    glottal = _REPO / f"features_{lang}_QCP_python_DDK.csv"
    return hubert, whisper, glottal


def main() -> int:
    parser = argparse.ArgumentParser(description="Late fusion DDK classification (Σw=1)")
    parser.add_argument(
        "--presets",
        default="LF-H+W+G,LF-H+W-FT+G,LF-H+W-CTC+G",
        help="Comma-separated fusion presets",
    )
    parser.add_argument("--models", default="rf,xgb")
    parser.add_argument("--paper_min_sens", type=float, default=0.9)
    parser.add_argument("--weight_step", type=float, default=0.1)
    parser.add_argument("--output_run", default="", help="Run dir under classification_results/")
    args = parser.parse_args()

    preset_names = [p.strip() for p in args.presets.split(",") if p.strip()]
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_run) if args.output_run else _REPO / "classification_results" / f"run_late_fusion_{ts}"
    if not run_dir.is_absolute():
        run_dir = _REPO / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []
    all_folds: List[pd.DataFrame] = []

    for preset_name in preset_names:
        if preset_name not in FUSION_PRESETS:
            print(f"[warn] Unknown preset {preset_name!r}, skip")
            continue
        preset = FUSION_PRESETS[preset_name]

        for lang in LANGS:
            hubert_p, whisper_p, glottal_p = _paths_for_lang(lang, preset)
            missing = [str(p) for p in (hubert_p, whisper_p, glottal_p) if not p.is_file()]
            if missing:
                print(f"[skip] {preset_name} {lang}: missing {missing}")
                continue

            merged, mod_cols = merge_modality_tables(hubert_p, whisper_p, glottal_p)
            csv_tag = f"features_{lang}_{preset.whisper_csv_suffix}_latefusion.csv"
            task_stem = Path(csv_tag).stem
            task_out = run_dir / task_stem
            task_out.mkdir(parents=True, exist_ok=True)

            for model_key in model_keys:
                sample_res, spk_res, fold_df = evaluate_late_fusion(
                    merged,
                    mod_cols,
                    model_key=model_key,
                    fusion_name=preset_name,
                    csv_tag=csv_tag,
                    paper_min_sens=float(args.paper_min_sens),
                    weight_step=float(args.weight_step),
                )
                row = {
                    "csv_file": csv_tag,
                    "task_stem": task_stem,
                    "language": infer_language_code(csv_tag),
                    "task_type": "DDK",
                    "model": model_key,
                    "model_title": model_key.upper(),
                    "feature_subset": preset_name,
                    "n_features": sum(len(v) for v in mod_cols.values()),
                    "n_rows": len(merged),
                }
                row.update(compute_metrics_summary_row(sample_res, spk_res))
                row.update(compute_paper_metrics_summary_row(fold_df))
                summary_rows.append(row)
                all_folds.append(fold_df)
                fold_df.to_csv(task_out / f"fold_metrics_{model_key}.csv", index=False)

    if not summary_rows:
        print("No late-fusion results produced (missing CSVs?).", flush=True)
        return 1

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "global_summary_metrics.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(run_dir / "all_fold_metrics.csv", index=False)

    link = _REPO / "classification_results" / "run_late_fusion_ddk"
    if link.is_symlink():
        link.unlink()
    elif link.exists() and link.is_dir() and not any(link.iterdir()):
        link.rmdir()
    if not link.exists():
        link.symlink_to(run_dir.name, target_is_directory=True)

    print(f"\nLate fusion complete: {run_dir}")
    print(f"Symlink: classification_results/run_late_fusion_ddk")
    print("Regenerate tables: python3 build_ddk_f1_tables.py --write-latex --write-md --dual-level")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
