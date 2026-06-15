#!/usr/bin/env python3
"""
Prepare Bridge2AI NORM data for SLAM-LLM Whisper CTC fine-tuning.

Reads a Zipformer manifest (feature_csv,transcript), e.g. train.norm.txt, maps each
row to an existing reconstructed wav via subset_meta.tsv, and writes:

  <output_dir>/wav/<utt_id>.wav          (symlinks by default)
  <output_dir>/jsonl/train.jsonl         SLAM-LLM format: {"source": "...", "target": "..."}
  <output_dir>/jsonl/dev.jsonl           from subset_meta dev split (normalized text)
  <output_dir>/jsonl/test.jsonl            from subset_meta test split (normalized text)

Example:
  python3 prepare_bridge2ai_norm_ctc_data.py \\
    --train-manifest /Users/ipatsoura/Downloads/train.norm.txt \\
    --output-dir bridge2ai_norm_ctc
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from whisper_bridge2ai_finetune import load_feature_to_wav_map, read_manifest


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def feature_to_utt_id(feature_csv: str) -> str:
    stem = Path(feature_csv).stem
    return stem.replace(" ", "_")


def symlink_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        import shutil

        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def write_jsonl(
    rows: Iterable[Tuple[Path, str]],
    out_path: Path,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for wav_path, target in rows:
            rec = {"source": str(wav_path.resolve()), "target": target}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def prepare_split_manifest(
    manifest_path: Path,
    feature_to_wav: Dict[str, str],
    repo_root: Path,
    wav_dir: Path,
    *,
    copy: bool,
) -> List[Tuple[Path, str]]:
    rows: List[Tuple[Path, str]] = []
    missing_map = 0
    missing_wav = 0
    for feat, txt in read_manifest(manifest_path):
        wav_rel = feature_to_wav.get(feat)
        if not wav_rel:
            missing_map += 1
            continue
        src = (repo_root / wav_rel).resolve()
        if not src.is_file():
            missing_wav += 1
            continue
        utt_id = feature_to_utt_id(feat)
        dst = wav_dir / f"{utt_id}.wav"
        symlink_or_copy(src, dst, copy=copy)
        rows.append((dst, normalize_text(txt)))
    if missing_map:
        print(f"[warn] {missing_map} rows missing from subset_meta: {manifest_path}")
    if missing_wav:
        print(f"[warn] {missing_wav} rows missing wav files: {manifest_path}")
    return rows


def prepare_meta_split(
    subset_meta_tsv: Path,
    split: str,
    repo_root: Path,
    wav_dir: Path,
    *,
    copy: bool,
) -> List[Tuple[Path, str]]:
    rows: List[Tuple[Path, str]] = []
    missing_wav = 0
    with subset_meta_tsv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if (row.get("split") or "").strip() != split:
                continue
            feat = (row.get("feature_csv") or "").strip()
            wav_rel = (row.get("wav_path") or "").strip()
            transcript = (row.get("transcript") or "").strip()
            if not feat or not wav_rel or not transcript:
                continue
            src = (repo_root / wav_rel).resolve()
            if not src.is_file():
                missing_wav += 1
                continue
            utt_id = feature_to_utt_id(feat)
            dst = wav_dir / f"{utt_id}.wav"
            symlink_or_copy(src, dst, copy=copy)
            rows.append((dst, normalize_text(transcript)))
    if missing_wav:
        print(f"[warn] {missing_wav} {split} rows missing wav files")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument(
        "--subset-meta-tsv",
        type=Path,
        default=Path("bridge2ai_zipformer_full_all_tasks/subset_meta.tsv"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("bridge2ai_norm_ctc"))
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy wav files instead of symlinking (for cluster transfer).",
    )
    p.add_argument("--skip-dev-test", action="store_true")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent
    output_dir = (repo_root / args.output_dir).resolve()
    wav_dir = output_dir / "wav"
    jsonl_dir = output_dir / "jsonl"
    feature_to_wav = load_feature_to_wav_map((repo_root / args.subset_meta_tsv).resolve())

    train_manifest = args.train_manifest.resolve()
    train_rows = prepare_split_manifest(
        train_manifest,
        feature_to_wav,
        repo_root,
        wav_dir,
        copy=args.copy,
    )
    n_train = write_jsonl(train_rows, jsonl_dir / "train.jsonl")

    n_dev = n_test = 0
    if not args.skip_dev_test:
        meta_tsv = (repo_root / args.subset_meta_tsv).resolve()
        dev_rows = prepare_meta_split(meta_tsv, "dev", repo_root, wav_dir, copy=args.copy)
        test_rows = prepare_meta_split(meta_tsv, "test", repo_root, wav_dir, copy=args.copy)
        n_dev = write_jsonl(dev_rows, jsonl_dir / "dev.jsonl")
        n_test = write_jsonl(test_rows, jsonl_dir / "test.jsonl")

    summary = {
        "train_manifest": str(train_manifest),
        "output_dir": str(output_dir),
        "wav_dir": str(wav_dir),
        "jsonl_dir": str(jsonl_dir),
        "train_utts": n_train,
        "dev_utts": n_dev,
        "test_utts": n_test,
        "mode": "copy" if args.copy else "symlink",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
