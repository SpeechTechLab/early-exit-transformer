import argparse
from pathlib import Path

import pandas as pd

from extract_glottal_features_qcp import extract_file_qcp


def _infer_task_from_path(wav_path: Path) -> str:
    # Keep it simple/consistent with other extractors: task comes from the folder name if present.
    for tok in wav_path.parts:
        t = tok.lower()
        if t in {"vowel", "vowels"}:
            return "vowels"
    return "vowels"


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

        print(f"[{idx}/{len(wavs)}] {wav}")
        row = extract_file_qcp(str(wav))
        if row is None:
            continue

        # If you have metadata (speaker/label), merge it later; this keeps extraction consistent across datasets.
        row["speaker"] = "unknown"
        row["label"] = "unknown"
        row["task"] = _infer_task_from_path(wav)
        rows.append(row)
        done.add(wav.stem)

        if save_every > 0 and len(rows) % save_every == 0:
            pd.DataFrame(rows).to_csv(output_csv, index=False)
            print(f"Checkpoint saved: {len(rows)} rows -> {output_csv}")

    if not rows:
        raise RuntimeError("No features extracted (all files failed?)")

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved {len(rows)} rows -> {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Extract QCP glottal features for a vowels dataset (Python/QCP).")
    parser.add_argument("--wav_root", type=str, required=True, help="Root directory containing .wav files")
    parser.add_argument("--output_csv", type=str, default="features_Vowels_QCP_python.csv")
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

