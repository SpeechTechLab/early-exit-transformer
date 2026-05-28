from __future__ import annotations

import argparse
from pathlib import Path

from extract_hubert_embeddings import run_extraction


def _infer_task_from_path(wav_path: Path) -> str:
    for tok in wav_path.parts:
        t = tok.lower()
        if t in {"vowel", "vowels"}:
            return "vowels"
    return "vowels"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract HuBERT embeddings for a vowels dataset")
    parser.add_argument("--wav_root", default="Vowels", help="Root directory containing .wav files")
    parser.add_argument("--output_csv", default="features_Vowels_hubert.csv")
    parser.add_argument("--model_id", default="facebook/hubert-base-ls960")
    parser.add_argument("--local_model_dir", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--pool", choices=["mean", "mean_std"], default="mean")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    wav_root = Path(args.wav_root)
    wavs = sorted(wav_root.rglob("*.wav"))
    if args.max_files and args.max_files > 0:
        wavs = wavs[: args.max_files]
    if not wavs:
        raise FileNotFoundError(f"No .wav files found under: {wav_root}")

    def metadata_fn(wav: Path) -> dict[str, str]:
        return {
            "speaker": "unknown",
            "label": "unknown",
            "task": _infer_task_from_path(wav),
        }

    run_extraction(
        wavs,
        Path(args.output_csv),
        metadata_fn,
        model_id=args.model_id,
        local_model_dir=args.local_model_dir or None,
        device=args.device or None,
        pool=args.pool,
        save_every=int(args.save_every),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
