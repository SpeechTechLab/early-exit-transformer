#!/usr/bin/env python3
"""Extract tuned Python glottal/QCP features for LibriSpeech train-clean-100.

This script reuses the tuned extractor implemented in `extract_glottal_features_qcp.py`
and builds one CSV row per utterance for LibriSpeech 100h.

Example:
    python3 extract_librispeech_100h_glottal_features.py \
        --librispeech_root LibriSpeech/train-clean-100 \
        --output_csv glottal_features_100h_QCP_python_tuned.csv \
        --num_workers 4 --save_every 100 --resume
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from extract_glottal_features_qcp import extract_file_qcp


def load_transcripts(librispeech_root: Path):
    """Load LibriSpeech transcripts into a dict keyed by utterance ID."""
    transcript_map = {}
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


def find_audio_files(librispeech_root: Path, extensions):
    files = []
    for ext in extensions:
        ext = ext if ext.startswith(".") else f".{ext}"
        files.extend(librispeech_root.rglob(f"*{ext}"))
    return sorted(set(p for p in files if p.is_file()))


def enrich_row(audio_path: Path, librispeech_root: Path, transcript_map, row: dict):
    """Attach LibriSpeech metadata to the extracted feature row."""
    utt_id = audio_path.stem
    try:
        rel = audio_path.relative_to(librispeech_root)
        parts = rel.parts
    except ValueError:
        parts = audio_path.parts

    speaker_id = parts[0] if len(parts) >= 1 else "unknown"
    chapter_id = parts[1] if len(parts) >= 2 else "unknown"

    row["file_name"] = utt_id
    row["speaker"] = str(speaker_id)
    row["label"] = "unknown"
    row["task"] = "LibriSpeech_train_clean_100"
    row["chapter_id"] = str(chapter_id)
    row["transcript"] = transcript_map.get(utt_id, "")
    row["path"] = str(audio_path)
    return row


def save_rows(rows, output_csv: Path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "file_name" in df.columns:
        df = df.sort_values("file_name").drop_duplicates(subset=["file_name"], keep="last")

    meta_cols = [
        "file_name",
        "speaker",
        "label",
        "task",
        "chapter_id",
        "transcript",
        "path",
    ]
    ordered_cols = [c for c in meta_cols if c in df.columns] + [c for c in df.columns if c not in meta_cols]
    df = df[ordered_cols]
    df.to_csv(output_csv, index=False)


def _extract_one(audio_path_str: str):
    return extract_file_qcp(audio_path_str)


def run(librispeech_root, output_csv, save_every=100, resume=False, max_files=0, num_workers=1, extensions=None):
    librispeech_root = Path(librispeech_root)
    output_csv = Path(output_csv)
    extensions = extensions or [".flac", ".wav"]

    if not librispeech_root.exists():
        raise FileNotFoundError(f"LibriSpeech root not found: {librispeech_root}")

    transcript_map = load_transcripts(librispeech_root)
    audio_files = find_audio_files(librispeech_root, extensions)
    if max_files and max_files > 0:
        audio_files = audio_files[:max_files]

    if not audio_files:
        raise RuntimeError(f"No audio files found under {librispeech_root} with extensions {extensions}")

    rows = []
    done = set()
    if resume and output_csv.exists():
        try:
            prev = pd.read_csv(output_csv)
            if "file_name" in prev.columns:
                done = set(prev["file_name"].astype(str))
                rows = prev.to_dict(orient="records")
                print(f"Resuming from {output_csv} with {len(done)} completed utterances")
        except Exception as exc:
            print(f"Warning: could not resume from {output_csv}: {exc}")

    pending = [p for p in audio_files if p.stem not in done]
    print(f"Found {len(audio_files)} audio files under {librispeech_root}")
    print(f"Loaded {len(transcript_map)} transcripts")
    print(f"Pending utterances: {len(pending)} | Workers: {num_workers}")

    if not pending:
        print("Nothing left to process.")
        save_rows(rows, output_csv)
        return

    start_time = time.time()
    processed_since_start = 0

    if num_workers <= 1:
        for idx, audio_path in enumerate(pending, start=1):
            print(f"[{idx}/{len(pending)}] {audio_path}")
            row = _extract_one(str(audio_path))
            if row is not None:
                rows.append(enrich_row(audio_path, librispeech_root, transcript_map, row))
                done.add(audio_path.stem)
            processed_since_start += 1

            if save_every > 0 and len(rows) % save_every == 0:
                save_rows(rows, output_csv)
                elapsed = time.time() - start_time
                rate = processed_since_start / max(elapsed, 1e-9)
                print(f"Checkpoint saved: {len(rows)} rows -> {output_csv} | rate={rate:.2f} utt/s")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            future_map = {ex.submit(_extract_one, str(audio_path)): audio_path for audio_path in pending}
            for idx, fut in enumerate(as_completed(future_map), start=1):
                audio_path = future_map[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    print(f"[{idx}/{len(pending)}] FAILED {audio_path}: {exc}")
                    continue

                if row is not None:
                    rows.append(enrich_row(audio_path, librispeech_root, transcript_map, row))
                    done.add(audio_path.stem)

                processed_since_start += 1
                if idx % 25 == 0 or idx == len(pending):
                    elapsed = time.time() - start_time
                    rate = processed_since_start / max(elapsed, 1e-9)
                    print(f"[{idx}/{len(pending)}] completed | rate={rate:.2f} utt/s")

                if save_every > 0 and len(rows) % save_every == 0:
                    save_rows(rows, output_csv)
                    print(f"Checkpoint saved: {len(rows)} rows -> {output_csv}")

    if not rows:
        raise RuntimeError("No features extracted; check the LibriSpeech root and audio files")

    save_rows(rows, output_csv)
    total_elapsed = time.time() - start_time
    print(f"Saved {len(rows)} rows to {output_csv}")
    print(f"Total elapsed: {total_elapsed / 60.0:.2f} minutes")


def main():
    parser = argparse.ArgumentParser(
        description="Extract tuned Python glottal/QCP features for LibriSpeech 100h"
    )
    parser.add_argument(
        "--librispeech_root",
        default="LibriSpeech/train-clean-100",
        help="Path to the LibriSpeech train-clean-100 directory",
    )
    parser.add_argument(
        "--output_csv",
        default="glottal_features_100h_QCP_python_tuned.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Checkpoint frequency in extracted utterances",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output CSV",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Optional limit for quick testing",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".flac", ".wav"],
        help="Audio file extensions to scan (default: .flac .wav)",
    )
    args = parser.parse_args()

    run(
        args.librispeech_root,
        args.output_csv,
        save_every=args.save_every,
        resume=args.resume,
        max_files=args.max_files,
        num_workers=args.num_workers,
        extensions=args.extensions,
    )


if __name__ == "__main__":
    main()
