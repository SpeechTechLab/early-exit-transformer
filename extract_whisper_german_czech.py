from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from extract_whisper_embeddings import run_extraction


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root_dir: Path
    metadata_csv: Path


def _load_speaker_metadata(metadata_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    for c in ["speaker_id", "group", "lang"]:
        if c not in df.columns:
            raise ValueError(f"metadata.csv missing required column '{c}': {metadata_csv}")
        df[c] = df[c].astype(str).str.strip()
    return df


def _group_to_label(group: str) -> str:
    g = str(group).strip().upper()
    if g == "PD":
        return "A"
    if g == "HC":
        return "C"
    return "UNKNOWN"


def _infer_speaker_id_from_filename(stem: str) -> Optional[str]:
    parts = stem.split("_")
    if len(parts) >= 3 and parts[1].upper() in {"HC", "PD"}:
        return "_".join(parts[:3])
    return None


def _infer_task_from_path(wav_path: Path) -> str:
    for tok in wav_path.parts:
        if tok.lower() in {"read", "ddk", "vowel", "vowels"}:
            return "vowel" if tok.lower() == "vowels" else tok.lower()
    return "unknown_task"


def extract_dataset(
    spec: DatasetSpec,
    output_csv: Path,
    *,
    model_id: str,
    local_model_dir: str | None,
    device: str | None,
    pool: str,
    max_files: int = 0,
    save_every: int = 50,
    resume: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    ddk_only: bool = False,
) -> None:
    meta = _load_speaker_metadata(spec.metadata_csv)
    meta = meta.set_index("speaker_id", drop=False)

    wavs = list(spec.root_dir.rglob("*.wav"))
    if ddk_only:
        wavs = [w for w in wavs if "ddk" in [p.lower() for p in w.parts]]
    wavs = sorted(wavs)
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(wavs)
    if max_files and max_files > 0:
        wavs = wavs[:max_files]

    if not wavs:
        raise RuntimeError(f"No .wav files found under {spec.root_dir}")

    def metadata_fn(wav: Path) -> dict[str, str]:
        speaker_id = _infer_speaker_id_from_filename(wav.stem)
        if speaker_id is None or speaker_id not in meta.index:
            return {"speaker": "unknown", "label": "UNKNOWN", "task": f"{spec.name}_unknown_task"}
        group = str(meta.loc[speaker_id, "group"])
        return {
            "speaker": speaker_id,
            "label": _group_to_label(group),
            "task": f"{spec.name}_{_infer_task_from_path(wav)}",
        }

    run_extraction(
        wavs,
        output_csv,
        metadata_fn,
        model_id=model_id,
        local_model_dir=local_model_dir,
        device=device,
        pool=pool,
        save_every=save_every,
        resume=resume,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Whisper encoder embeddings for German and Czech datasets"
    )
    parser.add_argument("--dataset", choices=["German", "Czech", "both"], default="both")
    parser.add_argument("--model_id", default="openai/whisper-large-v3")
    parser.add_argument("--local_model_dir", default="", help="Optional local model directory")
    parser.add_argument("--device", default="")
    parser.add_argument("--pool", choices=["mean", "mean_std"], default="mean")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=".")
    parser.add_argument(
        "--ddk_only",
        action="store_true",
        help="Only process wavs under a ddk/ subdirectory (skip read/vowel)",
    )
    parser.add_argument(
        "--output_tag",
        default="",
        help="Optional tag inserted in output CSV name, e.g. 'ft' -> features_German_whisper_ft_DDK.csv",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    tag = f"_{args.output_tag.strip()}" if str(args.output_tag).strip() else ""
    specs = {
        "German": DatasetSpec(name="German", root_dir=Path("German"), metadata_csv=Path("German/metadata.csv")),
        "Czech": DatasetSpec(name="Czech", root_dir=Path("Czech"), metadata_csv=Path("Czech/metadata.csv")),
    }

    targets = ["German", "Czech"] if args.dataset == "both" else [args.dataset]
    for ds in targets:
        spec = specs[ds]
        suffix = "_DDK" if args.ddk_only else ""
        out_csv = out_dir / f"features_{ds}_whisper{tag}{suffix}.csv"
        extract_dataset(
            spec=spec,
            output_csv=out_csv,
            model_id=args.model_id,
            local_model_dir=args.local_model_dir or None,
            device=args.device or None,
            pool=args.pool,
            max_files=args.max_files,
            save_every=args.save_every,
            resume=args.resume,
            shuffle=args.shuffle,
            seed=args.seed,
            ddk_only=bool(args.ddk_only),
        )


if __name__ == "__main__":
    main()
