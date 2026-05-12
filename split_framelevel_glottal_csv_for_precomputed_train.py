#!/usr/bin/env python3
"""Split a frame-level glottal CSV into per-utterance numeric CSVs + training manifest.

`train.py --use_precomputed_features` expects one line per utterance::

    /abs/path/to/<utt_id>.csv,transcript text

Each per-utterance CSV must be comma-separated floats only (no header), shape
``[n_frames, n_features]`` (or transposed); ``numpy.loadtxt`` loads it as in
``data.PrecomputedFeatureDataset``.

The frame-level extractor writes columns including ``file_name``, ``frame_idx``,
and the QCP keys in ``FRAME_FEATURE_KEYS``.

Typical workflow::

    # Optional but recommended: sort so utterances are contiguous (one streaming pass).
    (head -n1 glottal_features_train_clean_100_framelevel_QCP_python.csv \\
        && tail -n +2 glottal_features_train_clean_100_framelevel_QCP_python.csv \\
        | sort -t, -k1,1 -k2,2n -S 10G) > glottal_framelevel_sorted.csv

    python3 split_framelevel_glottal_csv_for_precomputed_train.py \\
        --frame_csv glottal_framelevel_sorted.csv \\
        --out_dir glottal_precomputed_train100h \\
        --manifest_out manifest_train_clean_100_glottal_only.txt \\
        --librispeech_root LibriSpeech/train-clean-100

If you skip sorting, this script still works but loads each ``file_name`` group
into memory the first time it sees that utterance (fine if the CSV is grouped
by ``file_name`` already; slow/high-memory if not).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from extract_glottal_features_qcp import FRAME_FEATURE_KEYS


def load_transcripts(librispeech_root: Path) -> dict[str, str]:
    transcript_map: dict[str, str] = {}
    for trans_path in sorted(librispeech_root.rglob("*.trans.txt")):
        with open(trans_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if not parts:
                    continue
                utt_id = parts[0]
                transcript = parts[1] if len(parts) > 1 else ""
                transcript_map[utt_id] = transcript
    return transcript_map


def _write_utt_matrix(out_dir: Path, utt_id: str, rows: list[dict], feature_keys: tuple[str, ...]) -> None:
    rows_sorted = sorted(rows, key=lambda r: int(r["frame_idx"]))
    mat = np.zeros((len(rows_sorted), len(feature_keys)), dtype=np.float64)
    for i, r in enumerate(rows_sorted):
        for j, k in enumerate(feature_keys):
            mat[i, j] = float(r[k])
    out_path = out_dir / f"{utt_id}.csv"
    np.savetxt(out_path, mat, delimiter=",", fmt="%.8e")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame_csv", type=str, required=True, help="Frame-level CSV (ideally sorted by file_name, frame_idx).")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for per-utterance <utt>.csv files.")
    parser.add_argument("--manifest_out", type=str, required=True, help="Output manifest path (one line per utt).")
    parser.add_argument(
        "--librispeech_root",
        type=str,
        required=True,
        help="Path to LibriSpeech/train-clean-100 (for *.trans.txt transcripts).",
    )
    parser.add_argument(
        "--skip_missing_transcript",
        action="store_true",
        help="Skip utterances with no transcript in librispeech_root (default: still write CSV, empty transcript in manifest).",
    )
    args = parser.parse_args()

    frame_csv = Path(args.frame_csv)
    out_dir = Path(args.out_dir)
    manifest_out = Path(args.manifest_out)
    ls_root = Path(args.librispeech_root)

    if not frame_csv.is_file():
        raise FileNotFoundError(frame_csv)
    if not ls_root.is_dir():
        raise FileNotFoundError(ls_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts = load_transcripts(ls_root)
    feature_keys = FRAME_FEATURE_KEYS

    current_utt: str | None = None
    buf: list[dict] = []
    written = 0
    skipped_no_trans = 0

    def flush_utt(utt_id: str, rows: list[dict]) -> None:
        nonlocal written, skipped_no_trans
        if not rows:
            return
        text = transcripts.get(utt_id, "")
        if args.skip_missing_transcript and not text.strip():
            skipped_no_trans += 1
            return
        _write_utt_matrix(out_dir, utt_id, rows, feature_keys)
        feat_path = out_dir.resolve() / f"{utt_id}.csv"
        with open(manifest_out, "a", encoding="utf-8") as mf:
            mf.write(f"{feat_path},{text}\n")
        written += 1

    if manifest_out.exists():
        manifest_out.unlink()

    with open(frame_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"file_name", "frame_idx", *feature_keys}
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        missing = need - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            utt = str(row["file_name"]).strip()
            if not utt:
                continue
            if current_utt is None:
                current_utt = utt
            if utt != current_utt:
                flush_utt(current_utt, buf)
                buf = []
                current_utt = utt
            buf.append(row)

        if current_utt is not None:
            flush_utt(current_utt, buf)

    print(f"Wrote {written} utterance feature files under {out_dir.resolve()}")
    print(f"Manifest: {manifest_out.resolve()} ({written} lines)")
    if skipped_no_trans:
        print(f"Skipped (no transcript): {skipped_no_trans}")


if __name__ == "__main__":
    main()
