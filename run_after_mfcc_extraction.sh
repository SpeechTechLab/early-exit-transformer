#!/usr/bin/env bash
# Watches for features_Vowels_QCP_python_mfcc.csv (1501 lines),
# then runs classification + final 3-way comparison.
set -e
cd "$(dirname "$0")"

TARGET_CSV="features_Vowels_QCP_python_mfcc.csv"
CLASSIFY_LOG="classification_vowels_qcp_python_mfcc.log"

echo "[watcher] Waiting for $TARGET_CSV to reach 1501 lines..."
while true; do
    if [ -f "$TARGET_CSV" ]; then
        LINES=$(wc -l < "$TARGET_CSV")
        echo "[watcher] $TARGET_CSV has $LINES lines"
        if [ "$LINES" -ge 1501 ]; then
            echo "[watcher] Extraction complete. Launching Python MFCC classification..."
            break
        fi
    fi
    sleep 120
done

python3 classify_tasks.py --csv "$TARGET_CSV" > "$CLASSIFY_LOG" 2>&1
echo "[watcher] Python MFCC classification done. Running 3-way comparison..."

python3 - <<'PY'
import pandas as pd
import glob
from pathlib import Path

results_dir = Path("classification_results")

def find_run(task_stem_pattern):
    """Return the most recent run directory containing a CSV matching task_stem_pattern."""
    csvs = sorted(results_dir.glob("run_*/global_summary_metrics.csv"), reverse=True)
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        if df['task_stem'].str.startswith(task_stem_pattern).any():
            return csv_path
    return None

# Locate each run's summary CSV
matlab_fixed_csv = find_run("features_Vowels_QCP")   # most recent MATLAB run (fixed code)
python_nomfcc_csv = find_run("features_Vowels_QCP_python")  # Python run without MFCC
python_mfcc_csv  = find_run("features_Vowels_QCP_python_mfcc")  # Python run with MFCC

if matlab_fixed_csv is None:
    raise FileNotFoundError("Could not find MATLAB fixed-code classification run")
if python_mfcc_csv is None:
    raise FileNotFoundError("Could not find Python MFCC classification run")

print(f"MATLAB fixed: {matlab_fixed_csv}")
print(f"Python no-MFCC: {python_nomfcc_csv}")
print(f"Python MFCC: {python_mfcc_csv}")

join_cols = ['model', 'feature_subset']
metric_cols = [
    'n_features',
    'speaker_auc_mean', 'speaker_pr_auc_mean',
    'speaker_f1_mean', 'speaker_balanced_acc_mean',
    'sample_auc_mean', 'sample_pr_auc_mean',
]

def load_run(csv_path, task_prefix, suffix):
    df = pd.read_csv(csv_path)
    df = df[df['task_stem'].str.startswith(task_prefix)].copy()
    rename = {c: f'{c}_{suffix}' for c in metric_cols if c in df.columns}
    return df[join_cols + [c for c in metric_cols if c in df.columns]].rename(columns=rename)

mat = load_run(matlab_fixed_csv,   "features_Vowels_QCP",        "mat")
py_mfcc = load_run(python_mfcc_csv, "features_Vowels_QCP_python_mfcc", "py_mfcc")

merged = mat.merge(py_mfcc, on=join_cols, how='outer')

# Add no-MFCC Python for comparison if available
if python_nomfcc_csv:
    py_nomfcc = load_run(python_nomfcc_csv, "features_Vowels_QCP_python", "py_nomfcc")
    merged = merged.merge(py_nomfcc, on=join_cols, how='outer')

# Compute MFCC gain: py_mfcc vs py_nomfcc for matching subsets
for base_col in ['speaker_auc_mean', 'speaker_pr_auc_mean', 'speaker_f1_mean']:
    if f'{base_col}_py_mfcc' in merged.columns and f'{base_col}_py_nomfcc' in merged.columns:
        merged[f'{base_col}_mfcc_gain'] = (
            merged[f'{base_col}_py_mfcc'] - merged[f'{base_col}_py_nomfcc']
        )
    if f'{base_col}_py_mfcc' in merged.columns and f'{base_col}_mat' in merged.columns:
        merged[f'{base_col}_py_vs_mat'] = (
            merged[f'{base_col}_py_mfcc'] - merged[f'{base_col}_mat']
        )

out = 'classification_results/three_way_comparison.csv'
merged.sort_values(join_cols).to_csv(out, index=False)
print(f"\nSaved 3-way comparison to {out}")
print("\n=== 3-WAY RESULTS (speaker AUC) ===")
print(merged.sort_values(join_cols)[[
    'model', 'feature_subset',
    'n_features_mat', 'speaker_auc_mean_mat',
    'n_features_py_mfcc', 'speaker_auc_mean_py_mfcc',
]].to_string(index=False))
PY

echo "[watcher] All done. See classification_results/three_way_comparison.csv"
