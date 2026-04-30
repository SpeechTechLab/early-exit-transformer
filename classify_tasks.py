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
from typing import Tuple
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


def compute_metrics_summary_row(results_sample: dict, results_speaker: dict):
    row = {}
    for metric in results_sample.keys():
        row[f"sample_{metric}_mean"] = float(np.nanmean(results_sample[metric]))
        row[f"sample_{metric}_std"] = float(np.nanstd(results_sample[metric]))
        row[f"speaker_{metric}_mean"] = float(np.nanmean(results_speaker[metric]))
        row[f"speaker_{metric}_std"] = float(np.nanstd(results_speaker[metric]))
    return row


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
                n_jobs=-1,
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
            })

            print(f"{model_name} | Fold {fold}/{outer_total} complete")

    return results_sample, results_speaker, pd.DataFrame(fold_records)


def build_feature_subsets(X: pd.DataFrame):
    glottal_base_roots = ['NAQ', 'QOQ', 'HRF', 'H1H2']
    direct_roots = ['G_RMS', 'G_ZCR', 'G_CREST', 'DG_PEAK', 'RES_RMS']

    glottal_base_cols = []
    direct_cols = []
    mfcc_cols = []

    for col in X.columns:
        if col.startswith('mfcc_'):
            mfcc_cols.append(col)
            continue

        if any(col.startswith(f"{r}_") for r in glottal_base_roots):
            glottal_base_cols.append(col)
        elif any(col.startswith(f"{r}_") for r in direct_roots):
            direct_cols.append(col)

    subsets = {
        'glottal_only': glottal_base_cols,
        'glottal_plus_direct': glottal_base_cols + direct_cols,
        'glottal_plus_direct_plus_mfcc': glottal_base_cols + direct_cols + mfcc_cols,
        'glottal_plus_mfcc': glottal_base_cols + mfcc_cols,
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
                "model": model_key,
                "model_title": model_cfg['title'],
                "feature_subset": subset_name,
                "n_features": len(subset_cols),
                "n_rows": len(data),
            }
            row.update(compute_metrics_summary_row(model_sample, model_speaker))
            summary_rows.append(row)

            if not fold_df.empty:
                fold_df["csv_file"] = csv_file
                fold_df["task_stem"] = task_stem
                fold_df["model"] = model_key
                fold_df["feature_subset"] = subset_name
                all_fold_dfs.append(fold_df)

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
    _args = _parser.parse_args()

    # Override global run config from CLI
    _models = [m.strip().lower() for m in str(_args.models).split(",") if m.strip()]
    if _models:
        RUN_MODELS[:] = _models

    globals()["OUTER_CV_MODE"] = str(_args.outer_cv).strip().lower()

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
        if summary_files:
            all_summary = pd.concat([pd.read_csv(f) for f in summary_files], ignore_index=True)
            all_summary.to_csv(run_output_dir / "global_summary_metrics.csv", index=False)

            # Convenience ranking by speaker AUC (descending)
            if "speaker_auc_mean" in all_summary.columns:
                ranked = all_summary.sort_values(by="speaker_auc_mean", ascending=False)
                ranked.to_csv(run_output_dir / "global_summary_ranked_by_speaker_auc.csv", index=False)

        fold_files = sorted(run_output_dir.glob("*/fold_metrics.csv"))
        if fold_files:
            all_folds = pd.concat([pd.read_csv(f) for f in fold_files], ignore_index=True)
            all_folds.to_csv(run_output_dir / "global_fold_metrics.csv", index=False)

        print(f"Global summary files saved in: {run_output_dir}")
