import argparse
from pathlib import Path

import pandas as pd

from extract_glottal_features_qcp import extract_file_qcp


def _infer_label_from_path(p: Path) -> str:
    # Convention used by classify_tasks.py: label A=1, C=0
    # Here we map PD -> A (affected), HC -> C (control).
    low_parts = [x.lower() for x in p.parts]
    if "pd" in low_parts:
        return "A"
    if "hc" in low_parts:
        return "C"
    return "UNKNOWN"


def _infer_task_from_path(p: Path) -> str:
    # Top-level DDK task folders observed in: DDK analysis/<task>/sin normalizar/<hc|pd>/...
    for tok in p.parts:
        t = tok.lower()
        if t in {"pataka", "pakata", "petaka", "pa-pa-pa", "ta-ta-ta", "ka-ka-ka"}:
            return t
    return "unknown_task"


def _infer_speaker_from_filename(stem: str) -> str:
    # Example: AVPEPUDEA0001_pataka -> speaker "AVPEPUDEA0001"
    # Fallback to full stem if unexpected format.
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def run(
    wav_root: Path,
    output_csv: Path,
    *,
    save_every: int = 50,
    resume: bool = False,
):
    wavs = sorted(wav_root.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found under: {wav_root}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    done: set[str] = set()

    if resume and output_csv.exists():
        prev = pd.read_csv(output_csv)
        if "file_name" in prev.columns:
            done = set(prev["file_name"].astype(str))
            rows = prev.to_dict(orient="records")
            print(f"Resuming from {output_csv} with {len(done)} completed files")

    for idx, wav in enumerate(wavs, start=1):
        if wav.stem in done:
            continue

        label = _infer_label_from_path(wav)
        task = _infer_task_from_path(wav)
        speaker = _infer_speaker_from_filename(wav.stem)

        print(f"[{idx}/{len(wavs)}] {wav} | speaker={speaker} label={label} task={task}")

        row = extract_file_qcp(str(wav))
        if row is None:
            continue

        row["speaker"] = speaker
        row["label"] = label
        row["task"] = f"Colombian_{task}"
        rows.append(row)
        done.add(wav.stem)

        if save_every > 0 and len(rows) % save_every == 0:
            pd.DataFrame(rows).to_csv(output_csv, index=False)
            print(f"Checkpoint saved: {len(rows)} rows -> {output_csv}")

    if not rows:
        raise RuntimeError("No features extracted (all files failed?)")

    df = pd.DataFrame(rows)
    meta_cols = ["file_name", "speaker", "label", "task"]
    other_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + other_cols]
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows -> {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract QCP glottal features for Colombian DDK wavs (pa-ta-ka etc.)"
    )
    parser.add_argument(
        "--wav_root",
        type=str,
        default="DDK analysis",
        help="Root directory containing the Colombian DDK .wav files",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="features_Colombian_QCP_python_DDK.csv",
        help="Output CSV path",
    )
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run(
        wav_root=Path(args.wav_root),
        output_csv=Path(args.output_csv),
        save_every=int(args.save_every),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()

