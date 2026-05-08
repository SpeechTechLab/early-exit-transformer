import pandas as pd
from pathlib import Path

mat = pd.read_csv('classification_results/run_20260403_100451/global_summary_metrics.csv')
py = pd.read_csv('classification_results/run_20260403_104736/global_summary_metrics.csv')

mat = mat[mat['task_stem'].eq('features_Vowels_QCP')].copy()
py = py[py['task_stem'].eq('features_Vowels_QCP_python')].copy()

join_cols = ['model', 'feature_subset']
cols = [
    'speaker_auc_mean','speaker_pr_auc_mean','speaker_f1_mean','speaker_balanced_acc_mean',
    'sample_auc_mean','sample_pr_auc_mean','sample_f1_mean','sample_balanced_acc_mean'
]
mat2 = mat[join_cols + cols].rename(columns={c: f'{c}_matlab' for c in cols})
py2 = py[join_cols + cols].rename(columns={c: f'{c}_python' for c in cols})
merged = mat2.merge(py2, on=join_cols, how='outer')
for c in cols:
    merged[f'{c}_delta_python_minus_matlab'] = merged[f'{c}_python'] - merged[f'{c}_matlab']

out='classification_results/python_vs_matlab_vowels_qcp_comparison.csv'
merged.sort_values(join_cols).to_csv(out, index=False)
print('saved', out)
print(merged.sort_values(join_cols).to_string(index=False))
