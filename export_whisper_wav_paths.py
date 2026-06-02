#!/usr/bin/env python3
"""List wav paths required for Whisper from Zipformer manifests + subset_meta.tsv.

Usage:
  python3 export_whisper_wav_paths.py \\
    --manifest bridge2ai_zipformer_full_all_tasks/manifests/clean_short_en/read_train.txt \\
    --manifest bridge2ai_zipformer_full_all_tasks/manifests/clean_short_en/read_dev.txt \\
    --output bridge2ai_zipformer_full_all_tasks/manifests/clean_short_en_wav_paths.txt

Transfer to cluster (from Mac, if wavs live under repo root):
  COPYFILE_DISABLE=1 tar czf - -T bridge2ai_zipformer_full_all_tasks/manifests/clean_short_en_wav_paths.txt \\
    | ssh -J user@jump host 'docker exec -i CONTAINER tar -xzf - -C /stek/patsoura/early-exit-transformer'
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from whisper_bridge2ai_finetune import load_feature_to_wav_map, read_manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument(
        "--subset_meta_tsv",
        type=Path,
        default=Path("bridge2ai_zipformer_full_all_tasks/subset_meta.tsv"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write one wav path per line (default: stdout).",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent
    feature_to_wav = load_feature_to_wav_map((repo_root / args.subset_meta_tsv).resolve())

    wavs: set[str] = set()
    missing_map = 0
    for manifest in args.manifest:
        for feat, _txt in read_manifest((repo_root / manifest).resolve()):
            wav = feature_to_wav.get(feat)
            if wav:
                wavs.add(wav)
            else:
                missing_map += 1

    lines = sorted(wavs)
    text = "\n".join(lines) + ("\n" if lines else "")
    if args.output:
        out = repo_root / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {len(lines)} wav paths to {out}")
    else:
        print(text, end="")
    if missing_map:
        print(f"[warn] {missing_map} manifest rows not in subset_meta.tsv", flush=True)


if __name__ == "__main__":
    main()
