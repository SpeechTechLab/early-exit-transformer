import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Optional

import pandas as pd

from extract_glottal_features_qcp import extract_file_qcp


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root_dir: Path
    metadata_csv: Path


def _load_speaker_metadata(metadata_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    # Normalize expected columns
    for c in ["speaker_id", "group", "lang"]:
        if c not in df.columns:
            raise ValueError(f"metadata.csv missing required column '{c}': {metadata_csv}")
        df[c] = df[c].astype(str).str.strip()
    return df


def _group_to_label(group: str) -> str:
    g = str(group).strip().upper()
    # Convention used by classify_tasks.py: label A=1, C=0
    # Here we map PD -> A (affected), HC -> C (control).
    if g == "PD":
        return "A"
    if g == "HC":
        return "C"
    return "UNKNOWN"


def _infer_speaker_id_from_filename(stem: str) -> Optional[str]:
    # Expected: deu_HC_001_read / cze_PD_003_ddk / etc.
    parts = stem.split("_")
    if len(parts) >= 3 and parts[1].upper() in {"HC", "PD"}:
        return "_".join(parts[:3])
    return None


def _infer_task_from_path(wav_path: Path) -> str:
    # e.g. German/read/HC/deu_HC_001_read.wav -> "read"
    #      Czech/ddk/PD/cze_PD_003_ddk.wav -> "ddk"
    for tok in wav_path.parts:
        if tok.lower() in {"read", "ddk", "vowel", "vowels"}:
            return "vowel" if tok.lower() == "vowels" else tok.lower()
    return "unknown_task"


def extract_dataset(
    spec: DatasetSpec,
    output_csv: Path,
    max_files: int = 0,
    save_every: int = 50,
    resume: bool = False,
    shuffle: bool = False,
    seed: int = 42,
):
    meta = _load_speaker_metadata(spec.metadata_csv)
    meta = meta.set_index("speaker_id", drop=False)

    wavs = list(spec.root_dir.rglob("*.wav"))
    wavs = sorted(wavs)
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(wavs)
    if max_files and max_files > 0:
        wavs = wavs[:max_files]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    done: set[str] = set()

    if resume and output_csv.exists():
        try:
            prev = pd.read_csv(output_csv)
            if "file_name" in prev.columns:
                done = set(prev["file_name"].astype(str))
                rows = prev.to_dict(orient="records")
                print(f"Resuming from {output_csv} with {len(done)} completed files")
        except Exception as exc:
            print(f"Warning: could not resume from {output_csv}: {exc}")

    if not wavs:
        raise RuntimeError(f"No .wav files found under {spec.root_dir}")

    for idx, wav in enumerate(wavs, start=1):
        if wav.stem in done:
            continue

        speaker_id = _infer_speaker_id_from_filename(wav.stem)
        if speaker_id is None:
            # Fall back: try parent dirs (HC/PD) won't include ID; skip safely.
            print(f"[{idx}/{len(wavs)}] Skipping (cannot infer speaker_id): {wav}")
            continue

        if speaker_id not in meta.index:
            print(f"[{idx}/{len(wavs)}] Skipping (speaker_id not in metadata): {wav}")
            continue

        group = str(meta.loc[speaker_id, "group"])
        label = _group_to_label(group)
        task = _infer_task_from_path(wav)

        print(f"[{idx}/{len(wavs)}] {wav} | speaker={speaker_id} group={group} task={task}")
        row = extract_file_qcp(str(wav))
        if row is None:
            continue

        row["speaker"] = speaker_id
        row["label"] = label
        row["task"] = f"{spec.name}_{task}"
        rows.append(row)
        done.add(wav.stem)

        if save_every > 0 and len(rows) % save_every == 0:
            df_ckpt = pd.DataFrame(rows)
            meta_cols = ["file_name", "speaker", "label", "task"]
            other_cols = [c for c in df_ckpt.columns if c not in meta_cols]
            df_ckpt = df_ckpt[meta_cols + other_cols]
            df_ckpt.to_csv(output_csv, index=False)
            print(f"Checkpoint saved: {len(df_ckpt)} rows -> {output_csv}")

    if not rows:
        raise RuntimeError(f"No features extracted for dataset {spec.name}; check filenames + metadata mapping")

    df = pd.DataFrame(rows)
    meta_cols = ["file_name", "speaker", "label", "task"]
    other_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + other_cols]
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows -> {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Extract QCP glottal features for German and Czech datasets")
    parser.add_argument("--dataset", choices=["German", "Czech", "both"], default="both")
    parser.add_argument("--max_files", type=int, default=0, help="Optional limit for quick tests")
    parser.add_argument("--save_every", type=int, default=50, help="Checkpoint frequency (rows)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output CSV(s)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle file order (useful with --max_files)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (when --shuffle is set)")
    parser.add_argument("--out_dir", default=".", help="Output directory for feature CSVs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    specs = {
        "German": DatasetSpec(name="German", root_dir=Path("German"), metadata_csv=Path("German/metadata.csv")),
        "Czech": DatasetSpec(name="Czech", root_dir=Path("Czech"), metadata_csv=Path("Czech/metadata.csv")),
    }

    targets = ["German", "Czech"] if args.dataset == "both" else [args.dataset]
    for ds in targets:
        spec = specs[ds]
        out_csv = out_dir / f"features_{ds}_QCP_python.csv"
        extract_dataset(
            spec=spec,
            output_csv=out_csv,
            max_files=args.max_files,
            save_every=args.save_every,
            resume=args.resume,
            shuffle=args.shuffle,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

