#!/usr/bin/env python3
"""Create minimal global_summary_metrics.csv for cluster-only classification runs.

Values were captured from cluster logs (run_20260605_100410, run_20260607_130745).
After real CSVs are pulled with scripts/pull_cluster_classification_results.sh,
those files replace these snapshots automatically (full CSVs include Acc and AUC).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "classification_results"

# csv, lang, model, subset, sample_mean, sample_std, paper_mean, paper_std
Row = tuple[str, str, str, str, float, float, float, float]


def _rows(run: str, entries: list[Row], force: bool = False) -> None:
    out_dir = RESULTS / run
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "global_summary_metrics.csv"
    if path.exists() and not force:
        print(f"skip (exists): {path}")
        return
    records = []
    for csv_file, language, model, feature_subset, sm, ss, pm, ps in entries:
        records.append(
            {
                "csv_file": csv_file,
                "language": language,
                "model": model,
                "feature_subset": feature_subset,
                "sample_f1_mean": sm,
                "sample_f1_std": ss,
                "paper_f1_mean": pm,
                "paper_f1_std": ps,
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"wrote {path}")


WHISPER_FT: list[Row] = [
    ("features_Czech_whisper_ft_DDK.csv", "CZ", "rf", "whisper_all", 0.607456, 0.138999, 0.707293, 0.085846),
    ("features_Czech_whisper_ft_DDK.csv", "CZ", "xgb", "whisper_all", 0.562190, 0.158686, 0.678032, 0.100269),
    ("features_Czech_whisper_ft_glottal_DDK.csv", "CZ", "rf", "whisper_plus_glottal_plus_direct", 0.646255, 0.116082, 0.689104, 0.075166),
    ("features_Czech_whisper_ft_glottal_DDK.csv", "CZ", "xgb", "whisper_plus_glottal_plus_direct", 0.604507, 0.132656, 0.703330, 0.087944),
    ("features_German_whisper_ft_DDK.csv", "DE", "rf", "whisper_all", 0.783740, 0.070639, 0.747884, 0.067657),
    ("features_German_whisper_ft_DDK.csv", "DE", "xgb", "whisper_all", 0.761630, 0.094408, 0.734702, 0.104850),
    ("features_German_whisper_ft_glottal_DDK.csv", "DE", "rf", "whisper_plus_glottal_plus_direct", 0.799493, 0.068452, 0.729904, 0.061146),
    ("features_German_whisper_ft_glottal_DDK.csv", "DE", "xgb", "whisper_plus_glottal_plus_direct", 0.797451, 0.077846, 0.766737, 0.098222),
    ("features_Colombian_whisper_ft_DDK.csv", "ES", "rf", "whisper_all", 0.754081, 0.092239, 0.794452, 0.145013),
    ("features_Colombian_whisper_ft_DDK.csv", "ES", "xgb", "whisper_all", 0.762062, 0.056235, 0.818392, 0.093000),
    ("features_Colombian_whisper_ft_glottal_DDK.csv", "ES", "rf", "whisper_plus_glottal_plus_direct", 0.756181, 0.105210, 0.819052, 0.118571),
    ("features_Colombian_whisper_ft_glottal_DDK.csv", "ES", "xgb", "whisper_plus_glottal_plus_direct", 0.795878, 0.070563, 0.881010, 0.080925),
]

TRIPLE_FT: list[Row] = [
    ("features_Czech_hubert_whisper_ft_glottal_DDK.csv", "CZ", "rf", "hubert_plus_whisper_plus_glottal_plus_direct", 0.6965, 0.1075, 0.747879, 0.113838),
    ("features_Czech_hubert_whisper_ft_glottal_DDK.csv", "CZ", "xgb", "hubert_plus_whisper_plus_glottal_plus_direct", 0.6919, 0.0881, 0.755000, 0.100200),
    ("features_German_hubert_whisper_ft_glottal_DDK.csv", "DE", "rf", "hubert_plus_whisper_plus_glottal_plus_direct", 0.7871, 0.0574, 0.751800, 0.045700),
    ("features_German_hubert_whisper_ft_glottal_DDK.csv", "DE", "xgb", "hubert_plus_whisper_plus_glottal_plus_direct", 0.8095, 0.0514, 0.778500, 0.052500),
    ("features_Colombian_hubert_whisper_ft_glottal_DDK.csv", "ES", "rf", "hubert_plus_whisper_plus_glottal_plus_direct", 0.7559, 0.0743, 0.827900, 0.114500),
    ("features_Colombian_hubert_whisper_ft_glottal_DDK.csv", "ES", "xgb", "hubert_plus_whisper_plus_glottal_plus_direct", 0.7843, 0.0664, 0.846100, 0.124300),
]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Overwrite existing snapshot CSVs")
    args = ap.parse_args()
    _rows("run_20260605_100410", WHISPER_FT, force=args.force)
    _rows("run_20260607_130745", TRIPLE_FT, force=args.force)


if __name__ == "__main__":
    main()
