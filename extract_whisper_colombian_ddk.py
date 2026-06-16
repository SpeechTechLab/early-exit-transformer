from __future__ import annotations

import argparse
from pathlib import Path

from extract_whisper_embeddings import run_extraction


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
        description="Extract Whisper encoder embeddings for Colombian DDK wavs (pa-ta-ka etc.)"
    )
    parser.add_argument("--wav_root", default="DDK analysis", help="Root directory with Colombian DDK wavs")
    parser.add_argument("--output_csv", default="features_Colombian_whisper_DDK.csv")
    parser.add_argument(
        "--output_tag",
        default="",
        help="If set, default output becomes features_Colombian_whisper_<tag>_DDK.csv",
    )
    parser.add_argument("--model_id", default="openai/whisper-large-v3")
    parser.add_argument("--local_model_dir", default="")
    parser.add_argument("--ctc_checkpoint", default="")
    parser.add_argument("--ctc_encoder_name", default="large-v3-turbo")
    parser.add_argument("--device", default="")
    parser.add_argument("--pool", choices=["mean", "mean_std"], default="mean")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_csv = args.output_csv
    if args.output_tag.strip() and output_csv == "features_Colombian_whisper_DDK.csv":
        output_csv = f"features_Colombian_whisper_{args.output_tag.strip()}_DDK.csv"

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
        Path(output_csv),
        metadata_fn,
        model_id=args.model_id,
        local_model_dir=args.local_model_dir or None,
        ctc_checkpoint=args.ctc_checkpoint or None,
        ctc_encoder_name=args.ctc_encoder_name,
        device=args.device or None,
        pool=args.pool,
        save_every=int(args.save_every),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
