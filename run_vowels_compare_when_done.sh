#!/usr/bin/env bash
set -e
while true; do
  lines=$(wc -l < features_Vowels_QCP_python.csv 2>/dev/null || echo 0)
  if [ "$lines" -ge 1501 ]; then
    break
  fi
  sleep 60
done
python3 compare_qcp_csvs.py --ref features_Vowels_QCP.csv --pred features_Vowels_QCP_python.csv --out qcp_compare_report_vowels_full.csv
python3 - <<'PY'
import pandas as pd
r = pd.read_csv('qcp_compare_report_vowels_full.csv')
with open('qcp_compare_vowels_full_summary.txt', 'w') as f:
    f.write(f"columns={len(r)}\n")
    f.write(f"mean_mae={r['mae'].mean():.6f}\n")
    f.write(f"mean_rmse={r['rmse'].mean():.6f}\n")
    f.write(f"mean_corr={r['corr'].mean():.6f}\n")
    f.write('worst_rmse_columns=' + ','.join(r.sort_values('rmse', ascending=False).head(10)['column'].tolist()) + '\n')
print('comparison summary saved')
PY
