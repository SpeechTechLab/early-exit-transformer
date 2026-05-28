#!/usr/bin/env python3
"""
Reconstruct approximate waveforms from Bridge2AI Voice parquet spectrograms.

Bridge2AI / PhysioNet processing (b2aiprep v3, torchaudio 2.8):
  - Audio resampled to 16 kHz (mono)
  - STFT: n_fft=400 (25 ms), win_length=400 (25 ms), hop_length=160 (10 ms)
  - Hann window, center=True (torchaudio default), power=2 spectrogram
  - Stored tensor: 10*log10(power) in dB, shape [201 freq x T]
  - Privacy: every 2nd time frame dropped (::2) → effective hop 320 samples (20 ms)

Inversion uses torchaudio.transforms.GriffinLim with the same STFT knobs when
available (recommended). Optional --undo-time-subsample upsamples the time
axis before inversion with hop=160 (original 10 ms) for slightly sharper audio.

There is no raw audio in the public release; reconstruction cannot be exact.

Example:
  python bridge2ai_spectrogram_to_waveform.py \\
    --dataset /Users/ipatsoura/Downloads/bridge2ai-voice-pediatric-dataset-1.1.0 \\
    --output-dir ./bridge2ai_pediatric_wav \\
    --limit 5 --undo-time-subsample --n-iter 64
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise SystemExit("librosa is required: pip install librosa") from exc

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyarrow is required: pip install pyarrow") from exc

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("soundfile is required: pip install soundfile") from exc

try:
    import torch
    import torchaudio

    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False


DEFAULT_DATASETS = {
    "pediatric": Path(
        "/Users/ipatsoura/Downloads/bridge2ai-voice-pediatric-dataset-1.1.0"
    ),
    "adult": Path(
        "/Users/ipatsoura/Downloads/bridge2ai-voice-an-ethically-sourced-diverse-voice-dataset-linked-to-health-information-3.1.0"
    ),
}

FEATURE_SPECS = {
    "spectrogram": {
        "parquet": "torchaudio_spectrogram.parquet",
        "json": "torchaudio_spectrogram.json",
        "column_candidates": ("spectrogram", "spectrograms"),
    },
    "mel": {
        "parquet": "torchaudio_mel_spectrogram.parquet",
        "json": "torchaudio_mel_spectrogram.json",
        "column_candidates": ("mel_spectrogram", "mel_spectrograms"),
    },
}


@dataclass(frozen=True)
class StftConfig:
    """STFT parameters aligned with Bridge2AI / PhysioNet documentation."""

    sample_rate: int = 16000
    n_fft: int = 400  # 25 ms @ 16 kHz
    win_length: int = 400  # 25 ms window
    hop_length_original: int = 160  # 10 ms hop (pre-privacy)
    hop_length_stored: int = 320  # after ::2 time subsampling (20 ms)
    n_mels: Optional[int] = None
    power: float = 2.0  # torchaudio Spectrogram power exponent
    center: bool = True  # torchaudio STFT center padding
    window: str = "hann"  # torchaudio default window_fn
    units: str = "dB"  # stored values are 10*log10(power)


def _find_tensor_field(meta: Dict[str, Any], column_candidates: Tuple[str, ...]) -> Dict[str, Any]:
    for field in meta.get("fields", []):
        if field.get("name") in column_candidates:
            return field
    raise ValueError(f"No field matching {column_candidates} in metadata JSON")


def load_stft_config(dataset_root: Path, feature_type: str) -> Tuple[StftConfig, Tuple[str, ...]]:
    spec = FEATURE_SPECS[feature_type]
    meta_path = dataset_root / "features" / spec["json"]
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    field = _find_tensor_field(meta, spec["column_candidates"])
    tensor = field["extras"]["tensor"]
    cfg = tensor.get("config", {})
    post = tensor.get("postprocessing", {}).get("subsample_time_axis", {}).get("effect", {})

    hop_orig = int(cfg.get("hop_length", 160))
    hop_stored = int(post.get("new_hop_length_samples", hop_orig))

    return (
        StftConfig(
            sample_rate=int(cfg.get("sample_rate", 16000)),
            n_fft=int(cfg.get("n_fft", 400)),
            win_length=int(cfg.get("win_length", 400)),
            hop_length_original=hop_orig,
            hop_length_stored=hop_stored,
            n_mels=int(cfg["n_mels"]) if "n_mels" in cfg else None,
            units=str(tensor.get("units", "dB")),
        ),
        spec["column_candidates"],
    )


def _resolve_column(batch_columns: List[str], candidates: Tuple[str, ...]) -> str:
    for name in candidates:
        if name in batch_columns:
            return name
    raise KeyError(f"None of {candidates} found in parquet columns: {batch_columns}")


def _safe_filename(participant_id: str, session_id: str, task_name: str) -> str:
    raw = f"{participant_id}__{session_id}__{task_name}"
    return re.sub(r"[^\w.\-]+", "_", raw).strip("_") + ".wav"


def db_power_spectrogram_to_power(spec_db: np.ndarray) -> np.ndarray:
    """Convert stored dB power spectrogram to linear power (torchaudio power=2)."""
    spec_db = spec_db.astype(np.float32)
    if np.nanmax(spec_db) < 50 and np.nanmin(spec_db) <= 0:
        # Typical stored log-power in dB
        return np.power(10.0, spec_db / 10.0, dtype=np.float32)
    # Fallback: treat as already linear power (PhysioNet README plotting path)
    return np.maximum(spec_db, 0.0).astype(np.float32)


def upsample_time_axis(spec: np.ndarray, factor: int = 2) -> np.ndarray:
    """Undo [::2] subsampling by linear interpolation along the time axis."""
    if factor <= 1:
        return spec
    n_freq, n_time = spec.shape
    new_t = n_time * factor
    if _HAS_TORCHAUDIO:
        x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
        y = torch.nn.functional.interpolate(
            x,
            size=(n_freq, new_t),
            mode="bilinear",
            align_corners=False,
        )
        return y.squeeze().cpu().numpy().astype(np.float32)
    x_old = np.linspace(0.0, 1.0, n_time, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, new_t, dtype=np.float32)
    out = np.empty((n_freq, new_t), dtype=np.float32)
    for i in range(n_freq):
        out[i] = np.interp(x_new, x_old, spec[i])
    return out


def _overlap_fraction(cfg: StftConfig, hop: int) -> float:
    return max(0.0, 1.0 - float(hop) / float(cfg.win_length))


def spectrogram_to_waveform(
    spec_db: np.ndarray,
    cfg: StftConfig,
    *,
    n_iter: int,
    method: str,
    undo_time_subsample: bool,
) -> np.ndarray:
    """
    Invert [freq, time] log-power spectrogram to mono waveform @ cfg.sample_rate.
    """
    if spec_db.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {spec_db.shape}")

    spec_power = db_power_spectrogram_to_power(spec_db)
    hop = cfg.hop_length_original if undo_time_subsample else cfg.hop_length_stored
    if undo_time_subsample and cfg.hop_length_stored == cfg.hop_length_original * 2:
        spec_power = upsample_time_axis(spec_power, factor=2)

    if method == "torchaudio_griffinlim":
        if not _HAS_TORCHAUDIO:
            raise RuntimeError("torchaudio not installed; use --method librosa_griffinlim")
        transform = torchaudio.transforms.GriffinLim(
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=hop,
            power=cfg.power,
            n_iter=n_iter,
            momentum=0.99,
            rand_init=True,
        )
        tensor = torch.from_numpy(spec_power).unsqueeze(0)
        return transform(tensor).squeeze(0).cpu().numpy()

    # librosa Griffin–Lim (Hann, centered STFT — closest librosa equivalent)
    return librosa.griffinlim(
        spec_power,
        n_iter=n_iter,
        n_fft=cfg.n_fft,
        hop_length=hop,
        win_length=cfg.win_length,
        window=cfg.window,
        center=cfg.center,
        momentum=0.99,
        random_state=0,
    )


def mel_to_waveform(
    mel: np.ndarray,
    cfg: StftConfig,
    *,
    n_iter: int,
    undo_time_subsample: bool,
) -> np.ndarray:
    if mel.ndim != 2:
        raise ValueError(f"Expected 2D mel spectrogram, got shape {mel.shape}")

    mel_f = mel.astype(np.float32)
    hop = cfg.hop_length_original if undo_time_subsample else cfg.hop_length_stored
    if undo_time_subsample and cfg.hop_length_stored == cfg.hop_length_original * 2:
        mel_f = upsample_time_axis(mel_f, factor=2)

    return librosa.feature.inverse.mel_to_audio(
        mel_f,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=hop,
        win_length=cfg.win_length,
        n_iter=n_iter,
        power=cfg.power,
    )


def iter_parquet_rows(
    parquet_path: Path,
    batch_size: int,
    column: str,
    limit: Optional[int],
) -> Iterator[Dict[str, Any]]:
    pf = pq.ParquetFile(parquet_path)
    produced = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        data = batch.to_pydict()
        n = len(data["participant_id"])
        for i in range(n):
            if limit is not None and produced >= limit:
                return
            yield {
                "participant_id": str(data["participant_id"][i]),
                "session_id": str(data["session_id"][i]),
                "task_name": str(data["task_name"][i]),
                "n_frames": int(data["n_frames"][i]) if "n_frames" in data else None,
                "features": np.asarray(data[column][i], dtype=np.float32),
            }
            produced += 1


def convert_dataset(
    dataset_root: Path,
    output_dir: Path,
    feature_type: str = "spectrogram",
    n_iter: int = 64,
    batch_size: int = 16,
    limit: Optional[int] = None,
    skip_existing: bool = True,
    dataset_label: Optional[str] = None,
    method: str = "torchaudio_griffinlim",
    undo_time_subsample: bool = False,
) -> int:
    dataset_root = dataset_root.resolve()
    spec = FEATURE_SPECS[feature_type]
    parquet_path = dataset_root / "features" / spec["parquet"]
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    cfg, column_candidates = load_stft_config(dataset_root, feature_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.tsv"
    write_manifest_header = not manifest_path.exists() or manifest_path.stat().st_size == 0

    pf = pq.ParquetFile(parquet_path)
    column = _resolve_column(pf.schema_arrow.names, column_candidates)
    total = pf.metadata.num_rows
    label = dataset_label or dataset_root.name

    hop_used = cfg.hop_length_original if undo_time_subsample else cfg.hop_length_stored
    print(f"\n[{label}] {feature_type} | rows={total} | column={column!r}")
    print(f"  parquet: {parquet_path}")
    print(f"  output:  {output_dir}")
    print(
        f"  PhysioNet STFT @ {cfg.sample_rate} Hz: "
        f"n_fft={cfg.n_fft} win={cfg.win_length} ({1000*cfg.win_length/cfg.sample_rate:.0f} ms) "
        f"hop_orig={cfg.hop_length_original} ({1000*cfg.hop_length_original/cfg.sample_rate:.0f} ms) "
        f"hop_used={hop_used} overlap={_overlap_fraction(cfg, hop_used):.0%} "
        f"center={cfg.center} window={cfg.window} power={cfg.power}"
    )
    print(
        f"  inversion: method={method} n_iter={n_iter} "
        f"undo_time_subsample={undo_time_subsample}"
    )

    written = 0
    skipped = 0
    manifest_lines: List[str] = []

    if write_manifest_header:
        manifest_lines.append(
            "participant_id\tsession_id\ttask_name\twav_path\tn_frames\t"
            "duration_sec\thop_length\tmethod\tundo_time_subsample\tfeature_type\n"
        )

    for row in iter_parquet_rows(parquet_path, batch_size, column, limit):
        fname = _safe_filename(row["participant_id"], row["session_id"], row["task_name"])
        out_path = output_dir / fname
        if skip_existing and out_path.is_file():
            skipped += 1
            continue

        features = row["features"]
        if feature_type == "spectrogram":
            audio = spectrogram_to_waveform(
                features,
                cfg,
                n_iter=n_iter,
                method=method,
                undo_time_subsample=undo_time_subsample,
            )
        else:
            audio = mel_to_waveform(
                features,
                cfg,
                n_iter=n_iter,
                undo_time_subsample=undo_time_subsample,
            )

        sf.write(out_path, audio, cfg.sample_rate)
        duration = len(audio) / cfg.sample_rate
        manifest_lines.append(
            f"{row['participant_id']}\t{row['session_id']}\t{row['task_name']}\t"
            f"{out_path}\t{row['n_frames'] or ''}\t{duration:.4f}\t{hop_used}\t"
            f"{method}\t{int(undo_time_subsample)}\t{feature_type}\n"
        )
        written += 1
        if written % 50 == 0:
            print(f"  written {written} (skipped {skipped}) ...", flush=True)

    if manifest_lines:
        with manifest_path.open("a", encoding="utf-8") as f:
            f.writelines(manifest_lines)

    print(f"  done: wrote {written}, skipped {skipped}, manifest={manifest_path}")
    return written


def parse_args() -> argparse.Namespace:
    default_method = "torchaudio_griffinlim" if _HAS_TORCHAUDIO else "librosa_griffinlim"
    parser = argparse.ArgumentParser(
        description="Reconstruct WAV from Bridge2AI spectrograms (PhysioNet STFT parameters).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, action="append")
    parser.add_argument("--both-default-datasets", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("./bridge2ai_waveforms"))
    parser.add_argument("--feature-type", choices=sorted(FEATURE_SPECS.keys()), default="spectrogram")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--method",
        choices=["torchaudio_griffinlim", "librosa_griffinlim"],
        default=default_method,
        help="Inversion backend (torchaudio matches b2aiprep / torchaudio 2.8).",
    )
    parser.add_argument(
        "--undo-time-subsample",
        action="store_true",
        help="Interpolate time axis 2× then invert with hop=160 (10 ms) before privacy ::2.",
    )
    parser.add_argument("--n-iter", type=int, default=64, help="Griffin–Lim iterations.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets: List[Tuple[str, Path]] = []

    if args.both_default_datasets:
        datasets.extend(DEFAULT_DATASETS.items())
    if args.dataset:
        for i, path in enumerate(args.dataset):
            datasets.append((path.name or f"dataset_{i}", path))

    if not datasets:
        print("Provide --dataset PATH and/or --both-default-datasets.", file=sys.stderr)
        sys.exit(2)

    if args.method == "torchaudio_griffinlim" and not _HAS_TORCHAUDIO:
        print("torchaudio not found; install torchaudio or use --method librosa_griffinlim", file=sys.stderr)
        sys.exit(1)

    skip_existing = not args.no_skip_existing
    total_written = 0

    for label, root in datasets:
        if not root.is_dir():
            print(f"Dataset not found: {root}", file=sys.stderr)
            sys.exit(1)
        out_dir = args.output_dir if args.output_dir and len(datasets) == 1 else args.output_root / label / args.feature_type
        total_written += convert_dataset(
            dataset_root=root,
            output_dir=out_dir,
            feature_type=args.feature_type,
            n_iter=args.n_iter,
            batch_size=args.batch_size,
            limit=args.limit,
            skip_existing=skip_existing,
            dataset_label=label,
            method=args.method,
            undo_time_subsample=args.undo_time_subsample,
        )

    print(f"\nTotal WAV files written: {total_written}")


if __name__ == "__main__":
    main()
