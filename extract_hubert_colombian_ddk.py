from __future__ import annotations

import argparse
from pathlib import Path

from extract_hubert_embeddings import run_extraction


def _infer_label_from_path(p: Path) -> str:
    low_parts = [x.lower() for x in p.parts]
    if "pd" in low_parts:
        return "A"
    if "hc" in low_parts:
        return "C"
    return "UNKNOWN"


def _infer_task_from_path(p: Path) -> str:
    for tok in p.parts:
        t = tok.lower()
        if t in {"pataka", "pakata", "petaka", "pa-pa-pa", "ta-ta-ta", "ka-ka-ka"}:
            return t
    return "unknown_task"


def _infer_speaker_from_filename(stem: str) -> str:
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract HuBERT embeddings for Colombian DDK wavs (pa-ta-ka etc.)"
    )
    parser.add_argument("--wav_root", default="DDK analysis", help="Root directory with Colombian DDK wavs")
    parser.add_argument("--output_csv", default="features_Colombian_hubert_DDK.csv")
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
            "speaker": _infer_speaker_from_filename(wav.stem),
            "label": _infer_label_from_path(wav),
            "task": f"Colombian_{_infer_task_from_path(wav)}",
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
