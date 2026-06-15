#!/usr/bin/env python3
"""List wav paths required for Bridge2AI NORM Whisper CTC prep.

Covers train.norm.txt plus dev/test rows from subset_meta.tsv (same as
prepare_bridge2ai_norm_ctc_data.py).

Usage:
  python3 export_norm_ctc_wav_paths.py \\
    --train-manifest bridge2ai_norm_ctc/manifests/train.norm.txt \\
    --output bridge2ai_norm_ctc/manifests/norm_ctc_wav_paths.txt

Transfer from Mac into Docker (paths relative to repo root):
  COPYFILE_DISABLE=1 tar czf - -T bridge2ai_norm_ctc/manifests/norm_ctc_wav_paths.txt \\
    | ssh -J ipatsoura@jumpsso.fbk.eu stek@digis-rf4421.fbk.eu \\
      'docker exec -i ee-ipatsoura-stek-4 tar -xzf - -C /stek/patsoura/early-exit-transformer'
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from whisper_bridge2ai_finetune import load_feature_to_wav_map, read_manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("bridge2ai_norm_ctc/manifests/train.norm.txt"),
    )
    p.add_argument(
        "--subset-meta-tsv",
        type=Path,
        default=Path("bridge2ai_zipformer_full_all_tasks/subset_meta.tsv"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("bridge2ai_norm_ctc/manifests/norm_ctc_wav_paths.txt"),
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent
    feature_to_wav = load_feature_to_wav_map((repo_root / args.subset_meta_tsv).resolve())

    wavs: set[str] = set()
    missing_map = 0

    for feat, _ in read_manifest((repo_root / args.train_manifest).resolve()):
        wav = feature_to_wav.get(feat)
        if wav:
            wavs.add(wav)
        else:
            missing_map += 1

    with (repo_root / args.subset_meta_tsv).open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if (row.get("split") or "").strip() not in {"dev", "test"}:
                continue
            wav = (row.get("wav_path") or "").strip()
            if wav:
                wavs.add(wav)

    out = repo_root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = sorted(wavs)
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Wrote {len(lines)} wav paths to {out}")
    if missing_map:
        print(f"[warn] {missing_map} train rows missing from subset_meta.tsv")


if __name__ == "__main__":
    main()
