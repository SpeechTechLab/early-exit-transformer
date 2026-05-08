#!/usr/bin/env python3
"""Filter a feature CSV by the `task` column and write a new CSV.

Intended for making DDK-only tables from `features_{Czech,German}_QCP_python.csv`
so results are comparable to DDK-only PD detection baselines.

Examples:
  python3 filter_feature_csv_by_task.py --in_csv features_German_QCP_python.csv --task_suffix _ddk
  python3 filter_feature_csv_by_task.py --in_csv features_Czech_QCP_python.csv  --task_suffix _ddk
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description="Filter a feature CSV by task")
    p.add_argument("--in_csv", required=True, help="Input feature CSV path")
    p.add_argument("--out_csv", default="", help="Output CSV path (default: <in>_filtered.csv)")
    p.add_argument("--task_suffix", default="", help="Keep only rows with task ending with this suffix (case-insensitive)")
    p.add_argument("--task_contains", default="", help="Keep only rows with task containing this substring (case-insensitive)")
    args = p.parse_args()

    in_csv = Path(args.in_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    if "task" not in df.columns:
        raise ValueError(f"CSV missing required column 'task': {in_csv}")

    suffix = str(args.task_suffix or "").strip()
    contains = str(args.task_contains or "").strip()
    if not suffix and not contains:
        raise ValueError("Provide at least one filter: --task_suffix or --task_contains")

    task = df["task"].astype(str).str.strip()
    mask = pd.Series(True, index=df.index)
    if suffix:
        mask &= task.str.lower().str.endswith(suffix.lower(), na=False)
    if contains:
        mask &= task.str.lower().str.contains(contains.lower(), na=False)

    out_csv = Path(args.out_csv) if args.out_csv else in_csv.with_name(in_csv.stem + "_filtered.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    kept = df[mask].copy()
    kept.to_csv(out_csv, index=False)
    kept_tasks = sorted(kept["task"].astype(str).str.strip().unique().tolist())
    print(f"{in_csv} -> {out_csv}")
    print(f"Rows kept: {len(kept)}/{len(df)}")
    print(f"Tasks kept: {kept_tasks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

