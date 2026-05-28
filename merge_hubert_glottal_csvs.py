#!/usr/bin/env python3
"""Merge HuBERT embedding CSVs with QCP glottal feature CSVs (one row per file_name)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

META_COLS = ["file_name", "speaker", "label", "task"]


def merge_pair(hubert_csv: Path, glottal_csv: Path, output_csv: Path) -> int:
    h = pd.read_csv(hubert_csv)
    g = pd.read_csv(glottal_csv)

    for col in META_COLS:
        if col not in h.columns or col not in g.columns:
            raise ValueError(f"Missing metadata column '{col}' in {hubert_csv} or {glottal_csv}")

    hubert_cols = [c for c in h.columns if c.startswith("hubert_")]
    if not hubert_cols:
        raise ValueError(f"No hubert_* columns in {hubert_csv}")

    glottal_cols = [c for c in g.columns if c not in META_COLS]
    if not glottal_cols:
        raise ValueError(f"No feature columns in {glottal_csv}")

    merged = h[META_COLS + hubert_cols].merge(
        g[["file_name"] + glottal_cols],
        on="file_name",
        how="inner",
    )

    if len(merged) == 0:
        raise RuntimeError(f"No overlapping file_name rows: {hubert_csv} × {glottal_csv}")

    only_h = set(h["file_name"].astype(str)) - set(merged["file_name"].astype(str))
    only_g = set(g["file_name"].astype(str)) - set(merged["file_name"].astype(str))
    if only_h:
        print(f"Warning: {len(only_h)} file(s) only in HuBERT CSV (dropped)")
    if only_g:
        print(f"Warning: {len(only_g)} file(s) only in glottal CSV (dropped)")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(
        f"{hubert_csv.name} + {glottal_csv.name} -> {output_csv.name} | "
        f"rows={len(merged)} | hubert={len(hubert_cols)} glottal={len(glottal_cols)}"
    )
    return len(merged)


# Default HuBERT + QCP pairs for this repo (DDK + vowels).
DEFAULT_PAIRS: list[tuple[str, str, str]] = [
    (
        "features_Czech_hubert_DDK.csv",
        "features_Czech_QCP_python_DDK.csv",
        "features_Czech_hubert_glottal_DDK.csv",
    ),
    (
        "features_German_hubert_DDK.csv",
        "features_German_QCP_python_DDK.csv",
        "features_German_hubert_glottal_DDK.csv",
    ),
    (
        "features_Colombian_hubert_DDK.csv",
        "features_Colombian_QCP_python_DDK.csv",
        "features_Colombian_hubert_glottal_DDK.csv",
    ),
    (
        "features_Czech_hubert.csv",
        "features_Czech_QCP_python.csv",
        "features_Czech_hubert_glottal.csv",
    ),
    (
        "features_German_hubert.csv",
        "features_German_QCP_python.csv",
        "features_German_hubert_glottal.csv",
    ),
    (
        "features_Vowels_hubert.csv",
        "features_Vowels_QCP_python.csv",
        "features_Vowels_hubert_glottal.csv",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge HuBERT and QCP glottal feature CSVs on file_name"
    )
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("HUBERT_CSV", "GLOTTAL_CSV", "OUT_CSV"),
        help="One triple: hubert csv, glottal csv, output csv (repeatable)",
    )
    parser.add_argument(
        "--defaults",
        action="store_true",
        help="Merge all standard DDK + vowels pairs (see DEFAULT_PAIRS in script)",
    )
    args = parser.parse_args()

    pairs = args.pair or []
    if args.defaults or not pairs:
        pairs.extend(DEFAULT_PAIRS)

    for hubert_path, glottal_path, out_path in pairs:
        merge_pair(Path(hubert_path), Path(glottal_path), Path(out_path))


if __name__ == "__main__":
    main()
