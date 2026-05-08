#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

TARGET_CSV="features_Vowels_QCP_python_mfcc_tuned.csv"
CLASSIFY_LOG="classification_vowels_qcp_python_mfcc_tuned.log"

echo "[watcher] Waiting for $TARGET_CSV to reach 1501 lines..."
while true; do
    if [ -f "$TARGET_CSV" ]; then
        LINES=$(wc -l < "$TARGET_CSV")
        echo "[watcher] $TARGET_CSV has $LINES lines"
        if [ "$LINES" -ge 1501 ]; then
            echo "[watcher] Extraction complete. Launching classification..."
            break
        fi
    fi
    sleep 120
done

python3 classify_tasks.py --csv "$TARGET_CSV" > "$CLASSIFY_LOG" 2>&1

echo "[watcher] Classification done. Building exact-match comparison..."
python3 - <<'PY'
import pandas as pd
from pathlib import Path

results_dir = Path('classification_results')

join_cols = ['model', 'feature_subset']
metric_cols = [
    'n_features',
    'speaker_auc_mean', 'speaker_pr_auc_mean', 'speaker_f1_mean', 'speaker_balanced_acc_mean',
    'sample_auc_mean', 'sample_pr_auc_mean', 'sample_f1_mean', 'sample_balanced_acc_mean'
]

def find_run_exact(task_stem):
    for csv_path in sorted(results_dir.glob('run_*/global_summary_metrics.csv'), reverse=True):
        df = pd.read_csv(csv_path)
        if df['task_stem'].eq(task_stem).any():
            return csv_path
    return None

def load_exact(csv_path, task_stem, suffix):
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    df = df[df['task_stem'].eq(task_stem)].copy()
    keep = [c for c in metric_cols if c in df.columns]
    rename = {c: f'{c}_{suffix}' for c in keep}
    return df[join_cols + keep].rename(columns=rename)

mat_csv = find_run_exact('features_Vowels_QCP')
py_nomfcc_csv = find_run_exact('features_Vowels_QCP_python')
py_old_mfcc_csv = find_run_exact('features_Vowels_QCP_python_mfcc')
py_tuned_csv = find_run_exact('features_Vowels_QCP_python_mfcc_tuned')

if py_tuned_csv is None:
    raise FileNotFoundError('Could not find tuned Python MFCC classification run')

merged = load_exact(mat_csv, 'features_Vowels_QCP', 'mat')
for csv_path, task_stem, suffix in [
    (py_nomfcc_csv, 'features_Vowels_QCP_python', 'py_nomfcc'),
    (py_old_mfcc_csv, 'features_Vowels_QCP_python_mfcc', 'py_old_mfcc'),
    (py_tuned_csv, 'features_Vowels_QCP_python_mfcc_tuned', 'py_tuned'),
]:
    part = load_exact(csv_path, task_stem, suffix)
    if part is not None:
        merged = merged.merge(part, on=join_cols, how='outer')

for base_col in ['speaker_auc_mean', 'speaker_pr_auc_mean', 'speaker_f1_mean']:
    if f'{base_col}_py_tuned' in merged.columns and f'{base_col}_mat' in merged.columns:
        merged[f'{base_col}_delta_tuned_minus_mat'] = merged[f'{base_col}_py_tuned'] - merged[f'{base_col}_mat']
    if f'{base_col}_py_tuned' in merged.columns and f'{base_col}_py_old_mfcc' in merged.columns:
        merged[f'{base_col}_delta_tuned_minus_old_mfcc'] = merged[f'{base_col}_py_tuned'] - merged[f'{base_col}_py_old_mfcc']

out = 'classification_results/tuned_mfcc_comparison.csv'
merged.sort_values(join_cols).to_csv(out, index=False)
print('saved', out)
print(merged.sort_values(join_cols)[[
    'model', 'feature_subset',
    'speaker_auc_mean_mat',
    'speaker_auc_mean_py_old_mfcc',
    'speaker_auc_mean_py_tuned'
]].to_string(index=False))
PY

echo "[watcher] Done. See classification_results/tuned_mfcc_comparison.csv"
