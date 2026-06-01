#!/usr/bin/env python3
"""
Filter Bridge2AI Zipformer manifests for ASR training.

Removes utterances that are poor training targets: missing mel CSVs, very short
transcripts, pediatric passage-8/9 fragments (single-word refs), and optional
task exclusions. Writes cleaned manifests and a short drop log.

Usage:
  python3 clean_bridge2ai_manifest.py \\
    --manifest bridge2ai_zipformer_full_all_tasks/manifests/read_train.txt \\
    --manifest bridge2ai_zipformer_full_all_tasks/manifests/read_dev.txt \\
    --manifest bridge2ai_zipformer_full_all_tasks/manifests/read_test.txt \\
    --in-place --backup-suffix .before_clean
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_UTT_RE = re.compile(r"^.+/(?P<cohort>adult|pediatric)_[^_]+_[^_]+_(?P<task>.+)\.csv$")


@dataclass
class Row:
    feat_path: str
    transcript: str
    cohort: str
    task: str


def _parse_line(line: str) -> Optional[Row]:
    line = line.strip()
    if not line:
        return None
    parts = line.split(",", 1)
    if len(parts) != 2:
        return None
    feat_path, transcript = parts[0].strip(), parts[1].strip()
    m = _UTT_RE.match(feat_path.replace("\\", "/"))
    if not m:
        return Row(feat_path, transcript, "", "")
    return Row(
        feat_path=feat_path,
        transcript=transcript,
        cohort=m.group("cohort"),
        task=m.group("task"),
    )


def _mel_time_frames(feat_path: Path, n_mels: int = 80) -> int:
    arr = np.loadtxt(feat_path, delimiter=",")
    if arr.ndim == 1:
        return 1
    if arr.shape[0] == n_mels:
        return int(arr.shape[1])
    if arr.shape[1] == n_mels:
        return int(arr.shape[0])
    return int(max(arr.shape))


def _should_keep(row: Row, args: argparse.Namespace) -> Tuple[bool, str]:
    if getattr(args, "dedupe_only", False):
        return True, ""
    feat = Path(row.feat_path)
    if not feat.is_file():
        return False, "missing_csv"

    words = row.transcript.split()
    n_words = len(words)
    n_chars = len(row.transcript)

    if n_chars < args.min_transcript_chars:
        return False, "short_chars"
    if n_words < args.min_words:
        return False, "short_words"
    if args.max_transcript_chars is not None and n_chars >= args.max_transcript_chars:
        return False, "long_transcript"

    if row.task in args.exclude_tasks:
        return False, f"exclude_task:{row.task}"

    for sub in args.exclude_task_substrings:
        if sub and sub in row.task:
            return False, f"exclude_substring:{sub}"

    if args.min_mel_frames > 0:
        try:
            n_frames = _mel_time_frames(feat, args.n_mels)
        except Exception:
            return False, "mel_read_error"
        if n_frames < args.min_mel_frames:
            return False, "short_mel"

    return True, ""


def _dedupe_passage_key(row: Row, args: argparse.Namespace) -> Optional[str]:
    if not args.dedupe_passage_transcript:
        return None
    if not row.task.startswith("passage"):
        return None
    return f"{row.task}\0{row.transcript}"


def _dedupe_rows(rows: List[Row], args: argparse.Namespace) -> Tuple[List[Row], dict]:
    """Drop duplicate feat paths and repeated passage (task, transcript) boilerplate."""
    drop_reasons: dict = {}
    kept: List[Row] = []
    seen_feat: set = set()
    seen_passage: set = set()

    for row in rows:
        feat_key = row.feat_path if args.dedupe_feat else None
        if feat_key is not None:
            if feat_key in seen_feat:
                drop_reasons["duplicate_feat"] = drop_reasons.get("duplicate_feat", 0) + 1
                continue
            seen_feat.add(feat_key)

        passage_key = _dedupe_passage_key(row, args)
        if passage_key is not None:
            if passage_key in seen_passage:
                drop_reasons["duplicate_passage_transcript"] = (
                    drop_reasons.get("duplicate_passage_transcript", 0) + 1
                )
                continue
            seen_passage.add(passage_key)

        kept.append(row)

    return kept, drop_reasons


def _filter_manifest(path: Path, args: argparse.Namespace) -> Tuple[int, int, dict]:
    rows: List[Row] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = _parse_line(line)
            if row is not None:
                rows.append(row)

    filtered: List[Row] = []
    drop_reasons: dict = {}
    for row in rows:
        ok, reason = _should_keep(row, args)
        if ok:
            filtered.append(row)
        else:
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    filtered, dedupe_reasons = _dedupe_rows(filtered, args)
    for reason, count in dedupe_reasons.items():
        drop_reasons[reason] = drop_reasons.get(reason, 0) + count

    kept = [f"{row.feat_path},{row.transcript}\n" for row in filtered]

    out_path = path
    if args.in_place and args.backup_suffix and not path.with_suffix(
        path.suffix + args.backup_suffix
    ).exists():
        shutil.copy2(path, path.with_suffix(path.suffix + args.backup_suffix))

    if args.output_dir:
        out_path = Path(args.output_dir) / path.name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(kept)

    return len(rows), len(kept), drop_reasons


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Manifest path (repeat for train/dev/test).",
    )
    p.add_argument("--in-place", action="store_true", help="Overwrite each manifest.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write cleaned manifests here (default: alongside input unless --in-place).",
    )
    p.add_argument(
        "--backup-suffix",
        type=str,
        default=".before_clean",
        help="When --in-place, copy original to manifest<suffix> once.",
    )
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument("--min-transcript-chars", type=int, default=8)
    p.add_argument("--max-transcript-chars", type=int, default=300)
    p.add_argument(
        "--min-mel-frames",
        type=int,
        default=100,
        help="Match train.py --min_mel_frames (0 to skip mel check).",
    )
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument(
        "--exclude-tasks",
        type=str,
        default="passage-8,passage-9",
        help="Comma-separated exact task names to drop.",
    )
    p.add_argument(
        "--exclude-task-substrings",
        type=str,
        default="",
        help="Drop if task name contains any of these substrings.",
    )
    p.add_argument(
        "--no-dedupe-feat",
        action="store_true",
        help="Keep duplicate feature paths if present.",
    )
    p.add_argument(
        "--no-dedupe-passage-transcript",
        action="store_true",
        help="Keep multiple speakers with the same passage task + transcript.",
    )
    p.add_argument(
        "--dedupe-only",
        action="store_true",
        help="Only dedupe; skip length/task/mel filters.",
    )
    args = p.parse_args()
    args.dedupe_feat = not args.no_dedupe_feat
    args.dedupe_passage_transcript = not args.no_dedupe_passage_transcript
    args.exclude_tasks = {s.strip() for s in args.exclude_tasks.split(",") if s.strip()}
    args.exclude_task_substrings = [
        s.strip() for s in args.exclude_task_substrings.split(",") if s.strip()
    ]

    if not args.in_place and args.output_dir is None:
        args.in_place = True

    total_in = total_out = 0
    for manifest in args.manifest:
        n_in, n_out, reasons = _filter_manifest(manifest, args)
        total_in += n_in
        total_out += n_out
        dropped = n_in - n_out
        print(f"{manifest}: {n_in} -> {n_out} (dropped {dropped})")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print(f"\nTotal: {total_in} -> {total_out} ({total_in - total_out} dropped)")


if __name__ == "__main__":
    main()
