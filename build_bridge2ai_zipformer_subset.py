#!/usr/bin/env python3
"""
Build a Bridge2AI Zipformer pilot subset (random duration cap) with precomputed mel CSVs only.

No glottal features are extracted or written. Joins bridge2ai_*_wav_v2/manifest.tsv with
features/static_features.tsv (transcription),
samples utterances until target seconds per cohort, holds out disjoint speakers for test,
extracts 80-dim mel frames matching train.py / util/data_loader.py, and writes manifests:

  feature_csv_path,transcript

Usage:
  python3 build_bridge2ai_zipformer_subset.py \\
    --output-dir bridge2ai_zipformer_pilot \\
    --adult-wav-manifest bridge2ai_adult_wav_v2/manifest.tsv \\
    --pediatric-wav-manifest bridge2ai_pediatric_wav_v2/manifest.tsv \\
    --adult-static-features /path/to/adult/features/static_features.tsv \\
    --pediatric-static-features /path/to/pediatric/features/static_features.tsv \\
    --adult-train-seconds 3600 \\
    --pediatric-train-seconds 1800
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


@dataclass(frozen=True)
class Utterance:
    cohort: str
    participant_id: str
    session_id: str
    task_name: str
    wav_path: Path
    duration_sec: float
    transcript: str

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.participant_id, self.session_id, self.task_name)

    @property
    def utt_id(self) -> str:
        return f"{self.cohort}_{self.participant_id}_{self.session_id}_{self.task_name}".replace(
            " ", "_"
        )


def _load_transcripts(static_features_path: Path) -> Dict[Tuple[str, str, str], str]:
    out: Dict[Tuple[str, str, str], str] = {}
    with static_features_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = (row.get("transcription") or "").strip()
            if not text or text.lower() in {"nan", "none"}:
                continue
            key = (row["participant_id"], row["session_id"], row["task_name"])
            out[key] = text
    return out


def _task_allowed(
    task_name: str,
    include_substrings: Optional[List[str]],
    exclude_substrings: List[str],
) -> bool:
    t = task_name.lower()
    for ex in exclude_substrings:
        if ex and ex in t:
            return False
    if not include_substrings:
        return True
    return any(inc in t for inc in include_substrings)


def _load_manifest_rows(
    cohort: str,
    manifest_path: Path,
    transcripts: Dict[Tuple[str, str, str], str],
    min_transcript_chars: int,
    max_transcript_chars: Optional[int],
    min_duration_sec: float,
    include_substrings: Optional[List[str]],
    exclude_substrings: List[str],
) -> List[Utterance]:
    pool: List[Utterance] = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            wav = Path(row["wav_path"])
            if not wav.is_file():
                continue
            key = (row["participant_id"], row["session_id"], row["task_name"])
            transcript = transcripts.get(key)
            if not transcript or len(transcript) < min_transcript_chars:
                continue
            if max_transcript_chars is not None and len(transcript) >= max_transcript_chars:
                continue
            if not _task_allowed(row["task_name"], include_substrings, exclude_substrings):
                continue
            try:
                duration = float(row["duration_sec"])
            except (TypeError, ValueError):
                duration = 0.0
            if duration <= 0 or duration < min_duration_sec:
                continue
            pool.append(
                Utterance(
                    cohort=cohort,
                    participant_id=row["participant_id"],
                    session_id=row["session_id"],
                    task_name=row["task_name"],
                    wav_path=wav,
                    duration_sec=duration,
                    transcript=transcript,
                )
            )
    return pool


def _split_speakers(
    pool: List[Utterance],
    test_speaker_fraction: float,
    rng: random.Random,
) -> Tuple[List[str], List[str]]:
    speakers = sorted({u.participant_id for u in pool})
    rng.shuffle(speakers)
    n_test = max(1, int(round(len(speakers) * test_speaker_fraction)))
    if len(speakers) <= 2:
        n_test = 1
    test_speakers = set(speakers[:n_test])
    train_speakers = set(speakers[n_test:])
    if not train_speakers:
        train_speakers = test_speakers
        test_speakers = set()
    return sorted(train_speakers), sorted(test_speakers)


def _split_dev_speakers(
    train_speakers: List[str],
    dev_speaker_fraction: float,
    rng: random.Random,
) -> Tuple[List[str], List[str]]:
    speakers = list(train_speakers)
    rng.shuffle(speakers)
    n_dev = max(1, int(round(len(speakers) * dev_speaker_fraction)))
    if len(speakers) <= 3:
        n_dev = 1
    dev_speakers = set(speakers[:n_dev])
    fit_speakers = set(speakers[n_dev:])
    if not fit_speakers:
        fit_speakers = dev_speakers
        dev_speakers = set()
    return sorted(fit_speakers), sorted(dev_speakers)


def _sample_until_seconds(
    candidates: List[Utterance],
    target_seconds: float,
    rng: random.Random,
) -> List[Utterance]:
    if target_seconds <= 0:
        return []
    rng.shuffle(candidates)
    picked: List[Utterance] = []
    total = 0.0
    for utt in candidates:
        if total >= target_seconds:
            break
        picked.append(utt)
        total += utt.duration_sec
    return picked


def _extract_mel_csv(
    wav_path: Path,
    out_csv: Path,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
) -> None:
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    spec_t = T.Spectrogram(
        n_fft=n_fft * 2, hop_length=hop_length, win_length=win_length
    )
    mel_t = T.MelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_fft + 1)
    spec = spec_t(waveform)
    mel = mel_t(spec).squeeze(0).numpy()  # [n_mels, time]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, mel, delimiter=",")


def _write_manifest(path: Path, rows: Iterable[Tuple[Path, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for feat_csv, transcript in rows:
            # Commas in transcript are allowed (split only on first comma in loader).
            line = f"{feat_csv},{transcript}\n"
            f.write(line)
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--adult-wav-manifest", type=Path, required=True)
    p.add_argument("--pediatric-wav-manifest", type=Path, required=True)
    p.add_argument("--adult-static-features", type=Path, required=True)
    p.add_argument("--pediatric-static-features", type=Path, required=True)
    p.add_argument("--adult-train-seconds", type=float, default=3600.0)
    p.add_argument("--pediatric-train-seconds", type=float, default=1800.0)
    p.add_argument(
        "--adult-test-seconds",
        type=float,
        default=None,
        help="Default: 15%% of adult train target.",
    )
    p.add_argument(
        "--pediatric-test-seconds",
        type=float,
        default=None,
        help="Default: 15%% of pediatric train target.",
    )
    p.add_argument("--test-speaker-fraction", type=float, default=0.15)
    p.add_argument(
        "--dev-speaker-fraction",
        type=float,
        default=0.10,
        help="Fraction of train speakers held out for dev (early-stop / checkpoint pick).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-transcript-chars", type=int, default=3)
    p.add_argument(
        "--min-duration-sec",
        type=float,
        default=1.0,
        help="Skip utterances shorter than this when building the pool (before sampling).",
    )
    p.add_argument(
        "--max-transcript-chars",
        type=int,
        default=300,
        help="Drop transcripts at or above this length (chars).",
    )
    p.add_argument(
        "--include-task-substrings",
        type=str,
        default="harvard-sentences,cape-v-sentences,rainbow-passage,caterpillar-passage",
        help="Comma-separated substrings; task must match at least one (empty = all tasks).",
    )
    p.add_argument(
        "--exclude-task-substrings",
        type=str,
        default="glides,maximum-phonation,prolonged-vowel,respiration-and-cough,diadochokinesis,loudness,picture-description,cinderella,story-recall,animal-fluency,silly-sounds,long-sounds",
        help="Comma-separated substrings; drop task if any match.",
    )
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-fft", type=int, default=512)
    p.add_argument("--hop-length", type=int, default=160)
    p.add_argument("--win-length", type=int, default=320)
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--skip-mel-extract", action="store_true")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = args.output_dir
    feat_dir = out_dir / "mels"

    adult_test_sec = (
        args.adult_test_seconds
        if args.adult_test_seconds is not None
        else 0.15 * args.adult_train_seconds
    )
    ped_test_sec = (
        args.pediatric_test_seconds
        if args.pediatric_test_seconds is not None
        else 0.15 * args.pediatric_train_seconds
    )

    cohorts = [
        ("adult", args.adult_wav_manifest, args.adult_static_features),
        ("pediatric", args.pediatric_wav_manifest, args.pediatric_static_features),
    ]

    include_substrings = [
        s.strip() for s in args.include_task_substrings.split(",") if s.strip()
    ] or None
    exclude_substrings = [
        s.strip() for s in args.exclude_task_substrings.split(",") if s.strip()
    ]

    train_utts: List[Utterance] = []
    dev_utts: List[Utterance] = []
    test_utts: List[Utterance] = []
    meta_rows: List[dict] = []

    for cohort, manifest_path, static_path in cohorts:
        transcripts = _load_transcripts(static_path)
        pool = _load_manifest_rows(
            cohort,
            manifest_path,
            transcripts,
            args.min_transcript_chars,
            args.max_transcript_chars,
            args.min_duration_sec,
            include_substrings,
            exclude_substrings,
        )
        train_speakers, test_speakers = _split_speakers(
            pool, args.test_speaker_fraction, rng
        )
        fit_speakers, dev_speakers = _split_dev_speakers(
            train_speakers, args.dev_speaker_fraction, rng
        )
        train_pool = [u for u in pool if u.participant_id in fit_speakers]
        dev_pool = [u for u in pool if u.participant_id in dev_speakers]
        test_pool = [u for u in pool if u.participant_id in test_speakers]

        if cohort == "adult":
            train_target, test_target = args.adult_train_seconds, adult_test_sec
        else:
            train_target, test_target = args.pediatric_train_seconds, ped_test_sec
        dev_target = 0.15 * train_target

        train_pick = _sample_until_seconds(train_pool, train_target, rng)
        dev_pick = _sample_until_seconds(dev_pool, dev_target, rng)
        test_pick = _sample_until_seconds(test_pool, test_target, rng)
        train_utts.extend(train_pick)
        dev_utts.extend(dev_pick)
        test_utts.extend(test_pick)

        train_dur = sum(u.duration_sec for u in train_pick)
        test_dur = sum(u.duration_sec for u in test_pick)
        dev_dur = sum(u.duration_sec for u in dev_pick)
        print(
            f"{cohort}: pool={len(pool)} utterances, "
            f"fit_speakers={len(fit_speakers)} dev_speakers={len(dev_speakers)} "
            f"test_speakers={len(test_speakers)}"
        )
        print(
            f"  train: {len(train_pick)} utts, {train_dur/3600:.3f} h "
            f"(target {train_target/3600:.3f} h)"
        )
        print(
            f"  dev:   {len(dev_pick)} utts, {dev_dur/3600:.3f} h "
            f"(target {dev_target/3600:.3f} h)"
        )
        print(
            f"  test:  {len(test_pick)} utts, {test_dur/3600:.3f} h "
            f"(target {test_target/3600:.3f} h)"
        )

    all_utts = train_utts + dev_utts + test_utts
    manifest_entries: List[Tuple[str, Path, str]] = []

    for utt in all_utts:
        feat_csv = feat_dir / f"{utt.utt_id}.csv"
        if not args.skip_mel_extract:
            _extract_mel_csv(
                utt.wav_path,
                feat_csv,
                sample_rate=args.sample_rate,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                n_mels=args.n_mels,
            )
        if utt in train_utts:
            split = "train"
        elif utt in dev_utts:
            split = "dev"
        else:
            split = "test"
        manifest_entries.append((split, feat_csv, utt.transcript))
        meta_rows.append(
            {
                "split": split,
                "cohort": utt.cohort,
                "participant_id": utt.participant_id,
                "session_id": utt.session_id,
                "task_name": utt.task_name,
                "wav_path": str(utt.wav_path),
                "feature_csv": str(feat_csv),
                "duration_sec": f"{utt.duration_sec:.4f}",
                "transcript": utt.transcript,
            }
        )

    train_rows = [(p, t) for s, p, t in manifest_entries if s == "train"]
    dev_rows = [(p, t) for s, p, t in manifest_entries if s == "dev"]
    test_rows = [(p, t) for s, p, t in manifest_entries if s == "test"]

    train_manifest = out_dir / "manifests" / "train.txt"
    dev_manifest = out_dir / "manifests" / "dev.txt"
    test_manifest = out_dir / "manifests" / "test.txt"
    n_train = _write_manifest(train_manifest, train_rows)
    n_dev = _write_manifest(dev_manifest, dev_rows)
    n_test = _write_manifest(test_manifest, test_rows)

    meta_path = out_dir / "subset_meta.tsv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(meta_rows)

    summary_path = out_dir / "README_subset.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"train_manifest={train_manifest} ({n_train} lines)",
                f"dev_manifest={dev_manifest} ({n_dev} lines)",
                f"test_manifest={test_manifest} ({n_test} lines)",
                f"feature_dim={args.n_mels} (mel only, no glottal)",
                f"seed={args.seed}",
                "",
                "Train (do NOT pass --append_glottal_features):",
                "  python3 train.py --use_precomputed_features \\",
                f"    --manifest {train_manifest} \\",
                f"    --n_glottal_features {args.n_mels} \\",
                "    --decoder_mode ctc --model_type early_zipformer_2layer_exits \\",
                "    --n_enc_layers_per_exit 2 --n_enc_exits 6 --n_epochs 30 \\",
                "    --batch_size 4 --n_workers 0 \\",
                f"    --save_model_dir {out_dir / 'trained_model'}",
                "",
                "Inference (held-out test manifest):",
                "  python3 inference.py --use_precomputed_features \\",
                f"    --manifest {test_manifest} \\",
                f"    --n_glottal_features {args.n_mels} \\",
                "    --decoder_mode ctc --model_type early_zipformer_2layer_exits \\",
                "    --n_enc_layers_per_exit 2 --n_enc_exits 6 \\",
                f"    --load_model_path {out_dir / 'trained_model'}/mod29-transformer \\",
                f"    --results_file {out_dir / 'wer_test.json'}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {train_manifest} ({n_train} utterances)")
    print(f"Wrote {dev_manifest} ({n_dev} utterances)")
    print(f"Wrote {test_manifest} ({n_test} utterances)")
    print(f"Meta: {meta_path}")
    print(f"Commands: {summary_path}")


if __name__ == "__main__":
    main()
