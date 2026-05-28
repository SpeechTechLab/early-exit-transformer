#!/usr/bin/env python3
"""Merge completed HuBERT+glottal classification runs (DDK + vowels) into one report folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from classify_tasks import _concat_nonempty_csvs, save_and_print_paper_style_reports


def combine_runs(
    ddk_run: Path,
    vowels_run: Path,
    output_run: Path,
    paper_min_sens: float = 0.9,
    paper_feature_subset: str = "hubert_plus_glottal_plus_direct",
) -> None:
    output_run.mkdir(parents=True, exist_ok=True)

    for src_run in (ddk_run, vowels_run):
        for task_dir in sorted(src_run.glob("features_*")):
            if not task_dir.is_dir():
                continue
            dest = output_run / task_dir.name
            if dest.exists():
                continue
            shutil.copytree(task_dir, dest)

    summary_files = sorted(output_run.glob("*/summary_metrics.csv"))
    all_summary = _concat_nonempty_csvs(summary_files)
    if all_summary.empty:
        raise SystemExit(f"No summary_metrics.csv under {output_run}")

    all_summary.to_csv(output_run / "global_summary_metrics.csv", index=False)
    if "speaker_auc_mean" in all_summary.columns:
        all_summary.sort_values("speaker_auc_mean", ascending=False).to_csv(
            output_run / "global_summary_ranked_by_speaker_auc.csv", index=False
        )

    fold_files = sorted(output_run.glob("*/fold_metrics.csv"))
    all_folds = _concat_nonempty_csvs(fold_files)
    if not all_folds.empty:
        all_folds.to_csv(output_run / "global_fold_metrics.csv", index=False)

    save_and_print_paper_style_reports(
        all_summary,
        output_run,
        min_sens=paper_min_sens,
        feature_subset=paper_feature_subset,
        models=["rf", "xgb"],
    )
    print(f"Combined report written to: {output_run}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddk-run",
        type=Path,
        default=Path("classification_results/run_20260521_114158"),
    )
    parser.add_argument(
        "--vowels-run",
        type=Path,
        default=Path("classification_results/run_20260521_181910"),
    )
    parser.add_argument(
        "--output-run",
        type=Path,
        default=Path("classification_results/run_hubert_glottal_combined"),
    )
    args = parser.parse_args()
    combine_runs(args.ddk_run.resolve(), args.vowels_run.resolve(), args.output_run.resolve())


if __name__ == "__main__":
    main()
