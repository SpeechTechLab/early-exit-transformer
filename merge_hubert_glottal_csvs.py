#!/usr/bin/env python3
"""Merge SSL embedding CSVs (HuBERT / Whisper) with QCP glottal feature CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

META_COLS = ["file_name", "speaker", "label", "task"]


def merge_pair(
    ssl_csv: Path,
    glottal_csv: Path,
    output_csv: Path,
    *,
    ssl_prefix: str = "hubert",
) -> int:
    h = pd.read_csv(ssl_csv)
    g = pd.read_csv(glottal_csv)

    for col in META_COLS:
        if col not in h.columns or col not in g.columns:
            raise ValueError(f"Missing metadata column '{col}' in {ssl_csv} or {glottal_csv}")

    ssl_cols = [c for c in h.columns if c.startswith(f"{ssl_prefix}_")]
    if not ssl_cols:
        raise ValueError(f"No {ssl_prefix}_* columns in {ssl_csv}")

    glottal_cols = [c for c in g.columns if c not in META_COLS]
    if not glottal_cols:
        raise ValueError(f"No feature columns in {glottal_csv}")

    merged = h[META_COLS + ssl_cols].merge(
        g[["file_name"] + glottal_cols],
        on="file_name",
        how="inner",
    )

    if len(merged) == 0:
        raise RuntimeError(f"No overlapping file_name rows: {ssl_csv} × {glottal_csv}")

    only_h = set(h["file_name"].astype(str)) - set(merged["file_name"].astype(str))
    only_g = set(g["file_name"].astype(str)) - set(merged["file_name"].astype(str))
    if only_h:
        print(f"Warning: {len(only_h)} file(s) only in {ssl_prefix} CSV (dropped)")
    if only_g:
        print(f"Warning: {len(only_g)} file(s) only in glottal CSV (dropped)")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(
        f"{ssl_csv.name} + {glottal_csv.name} -> {output_csv.name} | "
        f"rows={len(merged)} | {ssl_prefix}={len(ssl_cols)} glottal={len(glottal_cols)}"
    )
    return len(merged)


def merge_triple(
    hubert_csv: Path,
    whisper_csv: Path,
    glottal_csv: Path,
    output_csv: Path,
) -> int:
    """Merge HuBERT, Whisper, and glottal features on file_name."""
    h = pd.read_csv(hubert_csv)
    w = pd.read_csv(whisper_csv)
    g = pd.read_csv(glottal_csv)

    for col in META_COLS:
        if col not in h.columns or col not in w.columns or col not in g.columns:
            raise ValueError(
                f"Missing metadata column '{col}' in {hubert_csv}, {whisper_csv}, or {glottal_csv}"
            )

    hubert_cols = [c for c in h.columns if c.startswith("hubert_")]
    whisper_cols = [c for c in w.columns if c.startswith("whisper_")]
    glottal_cols = [c for c in g.columns if c not in META_COLS]
    if not hubert_cols:
        raise ValueError(f"No hubert_* columns in {hubert_csv}")
    if not whisper_cols:
        raise ValueError(f"No whisper_* columns in {whisper_csv}")
    if not glottal_cols:
        raise ValueError(f"No glottal feature columns in {glottal_csv}")

    merged = h[META_COLS + hubert_cols].merge(
        w[["file_name"] + whisper_cols],
        on="file_name",
        how="inner",
    ).merge(
        g[["file_name"] + glottal_cols],
        on="file_name",
        how="inner",
    )

    if len(merged) == 0:
        raise RuntimeError(
            f"No overlapping file_name rows: {hubert_csv} × {whisper_csv} × {glottal_csv}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(
        f"{hubert_csv.name} + {whisper_csv.name} + {glottal_csv.name} -> {output_csv.name} | "
        f"rows={len(merged)} | hubert={len(hubert_cols)} whisper={len(whisper_cols)} "
        f"glottal={len(glottal_cols)}"
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

DEFAULT_WHISPER_PAIRS: list[tuple[str, str, str]] = [
    (
        "features_Czech_whisper_DDK.csv",
        "features_Czech_QCP_python_DDK.csv",
        "features_Czech_whisper_glottal_DDK.csv",
    ),
    (
        "features_German_whisper_DDK.csv",
        "features_German_QCP_python_DDK.csv",
        "features_German_whisper_glottal_DDK.csv",
    ),
    (
        "features_Colombian_whisper_DDK.csv",
        "features_Colombian_QCP_python_DDK.csv",
        "features_Colombian_whisper_glottal_DDK.csv",
    ),
]

DEFAULT_TRIPLE_DDK_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "features_Czech_hubert_DDK.csv",
        "features_Czech_whisper_DDK.csv",
        "features_Czech_QCP_python_DDK.csv",
        "features_Czech_hubert_whisper_glottal_DDK.csv",
    ),
    (
        "features_German_hubert_DDK.csv",
        "features_German_whisper_DDK.csv",
        "features_German_QCP_python_DDK.csv",
        "features_German_hubert_whisper_glottal_DDK.csv",
    ),
    (
        "features_Colombian_hubert_DDK.csv",
        "features_Colombian_whisper_DDK.csv",
        "features_Colombian_QCP_python_DDK.csv",
        "features_Colombian_hubert_whisper_glottal_DDK.csv",
    ),
]

DEFAULT_TRIPLE_DDK_FT_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "features_Czech_hubert_DDK.csv",
        "features_Czech_whisper_ft_DDK.csv",
        "features_Czech_QCP_python_DDK.csv",
        "features_Czech_hubert_whisper_ft_glottal_DDK.csv",
    ),
    (
        "features_German_hubert_DDK.csv",
        "features_German_whisper_ft_DDK.csv",
        "features_German_QCP_python_DDK.csv",
        "features_German_hubert_whisper_ft_glottal_DDK.csv",
    ),
    (
        "features_Colombian_hubert_DDK.csv",
        "features_Colombian_whisper_ft_DDK.csv",
        "features_Colombian_QCP_python_DDK.csv",
        "features_Colombian_hubert_whisper_ft_glottal_DDK.csv",
    ),
]

DEFAULT_WHISPER_CTC_PAIRS: list[tuple[str, str, str]] = [
    (
        "features_Czech_whisper_ctc_DDK.csv",
        "features_Czech_QCP_python_DDK.csv",
        "features_Czech_whisper_ctc_glottal_DDK.csv",
    ),
    (
        "features_German_whisper_ctc_DDK.csv",
        "features_German_QCP_python_DDK.csv",
        "features_German_whisper_ctc_glottal_DDK.csv",
    ),
    (
        "features_Colombian_whisper_ctc_DDK.csv",
        "features_Colombian_QCP_python_DDK.csv",
        "features_Colombian_whisper_ctc_glottal_DDK.csv",
    ),
]

DEFAULT_WHISPER_FT_PAIRS: list[tuple[str, str, str]] = [
    (
        "features_Czech_whisper_ft_DDK.csv",
        "features_Czech_QCP_python_DDK.csv",
        "features_Czech_whisper_ft_glottal_DDK.csv",
    ),
    (
        "features_German_whisper_ft_DDK.csv",
        "features_German_QCP_python_DDK.csv",
        "features_German_whisper_ft_glottal_DDK.csv",
    ),
    (
        "features_Colombian_whisper_ft_DDK.csv",
        "features_Colombian_QCP_python_DDK.csv",
        "features_Colombian_whisper_ft_glottal_DDK.csv",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge SSL (HuBERT/Whisper) and QCP glottal feature CSVs on file_name"
    )
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("SSL_CSV", "GLOTTAL_CSV", "OUT_CSV"),
        help="One triple: ssl csv, glottal csv, output csv (repeatable)",
    )
    parser.add_argument(
        "--defaults",
        action="store_true",
        help="Merge all standard HuBERT DDK + vowels pairs (see DEFAULT_PAIRS in script)",
    )
    parser.add_argument(
        "--whisper_defaults",
        action="store_true",
        help="Merge all standard Whisper DDK pairs (see DEFAULT_WHISPER_PAIRS in script)",
    )
    parser.add_argument(
        "--whisper_ft_defaults",
        action="store_true",
        help="Merge all standard Whisper finetuned DDK pairs (see DEFAULT_WHISPER_FT_PAIRS)",
    )
    parser.add_argument(
        "--whisper_ctc_defaults",
        action="store_true",
        help="Merge all standard Whisper CTC-finetuned DDK pairs (see DEFAULT_WHISPER_CTC_PAIRS)",
    )
    parser.add_argument(
        "--triple",
        nargs=4,
        action="append",
        metavar=("HUBERT_CSV", "WHISPER_CSV", "GLOTTAL_CSV", "OUT_CSV"),
        help="Merge HuBERT + Whisper + glottal CSVs on file_name (repeatable)",
    )
    parser.add_argument(
        "--triple_defaults",
        action="store_true",
        help="Merge HuBERT + base Whisper + glottal for CZ/DE/CO DDK (DEFAULT_TRIPLE_DDK_PAIRS)",
    )
    parser.add_argument(
        "--triple_ft_defaults",
        action="store_true",
        help="Merge HuBERT + finetuned Whisper + glottal for DDK (DEFAULT_TRIPLE_DDK_FT_PAIRS)",
    )
    parser.add_argument(
        "--ssl_prefix",
        default="hubert",
        help="Embedding column prefix when using --pair (hubert or whisper)",
    )
    args = parser.parse_args()

    pairs = args.pair or []
    triples = args.triple or []
    ssl_prefix = str(args.ssl_prefix).strip() or "hubert"
    if args.defaults or (
        not pairs
        and not args.whisper_defaults
        and not args.whisper_ft_defaults
        and not args.whisper_ctc_defaults
        and not triples
        and not args.triple_defaults
        and not args.triple_ft_defaults
    ):
        pairs.extend(DEFAULT_PAIRS)
    if args.whisper_defaults:
        pairs.extend(DEFAULT_WHISPER_PAIRS)
        ssl_prefix = "whisper"
    if args.whisper_ft_defaults:
        pairs.extend(DEFAULT_WHISPER_FT_PAIRS)
        ssl_prefix = "whisper"
    if args.whisper_ctc_defaults:
        pairs.extend(DEFAULT_WHISPER_CTC_PAIRS)
        ssl_prefix = "whisper"
    if args.triple_defaults:
        triples.extend(DEFAULT_TRIPLE_DDK_PAIRS)
    if args.triple_ft_defaults:
        triples.extend(DEFAULT_TRIPLE_DDK_FT_PAIRS)

    for ssl_path, glottal_path, out_path in pairs:
        prefix = "whisper" if Path(ssl_path).stem.lower().find("whisper") >= 0 else ssl_prefix
        merge_pair(Path(ssl_path), Path(glottal_path), Path(out_path), ssl_prefix=prefix)

    for hubert_path, whisper_path, glottal_path, out_path in triples:
        merge_triple(
            Path(hubert_path),
            Path(whisper_path),
            Path(glottal_path),
            Path(out_path),
        )


if __name__ == "__main__":
    main()
