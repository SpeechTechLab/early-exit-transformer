"""Extract Whisper encoder embeddings for downstream classification.

Uses ``WhisperModel.encoder`` (final encoder layer / outer layer) and mean-pools
frame hidden states to one vector per wav file, matching the one-row-per-file QCP
CSV layout used by ``classify_tasks.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

DEFAULT_MODEL_ID = "openai/whisper-large-v3"
DEFAULT_FT_CHECKPOINT = "whisper_runs/bridge2ai_read_clean_short_en_ft_v2/checkpoint-150"
WHISPER_SAMPLE_RATE = 16000
_REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(_REPO_ROOT / ".hf_cache"))


def resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _checkpoint_architecture(checkpoint_dir: Path) -> str:
    cfg_path = checkpoint_dir / "config.json"
    if not cfg_path.exists():
        return ""
    import json

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    archs = cfg.get("architectures") or []
    return str(archs[0]) if archs else ""


def load_whisper(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    device: Optional[str] = None,
    local_model_dir: Optional[str] = None,
) -> tuple[torch.nn.Module, WhisperFeatureExtractor, str]:
    """Load Whisper encoder for embedding extraction.

    Supports base ``WhisperModel`` checkpoints and ASR-finetuned
    ``WhisperForConditionalGeneration`` Trainer checkpoints (encoder weights only).
    """
    dev = resolve_device(device)
    base_id = model_id

    if local_model_dir:
        ckpt = Path(local_model_dir)
        if not ckpt.is_dir():
            raise FileNotFoundError(f"Whisper checkpoint directory not found: {ckpt}")

        proc_src = str(ckpt) if (ckpt / "preprocessor_config.json").exists() else base_id
        processor = WhisperFeatureExtractor.from_pretrained(proc_src)

        arch = _checkpoint_architecture(ckpt)
        if "WhisperForConditionalGeneration" in arch:
            from transformers import WhisperForConditionalGeneration

            print(f"Loading finetuned Whisper encoder from {ckpt} ({arch})")
            full = WhisperForConditionalGeneration.from_pretrained(str(ckpt))
            model = full.model
        else:
            print(f"Loading Whisper encoder from {ckpt}")
            model = WhisperModel.from_pretrained(str(ckpt))
    else:
        processor = WhisperFeatureExtractor.from_pretrained(base_id)
        model = WhisperModel.from_pretrained(base_id)

    model = model.to(dev).eval()
    return model, processor, dev


def load_audio_for_whisper(file_path: str | Path) -> tuple[Optional[np.ndarray], Optional[int]]:
    """Load mono waveform resampled to 16 kHz for Whisper (no QCP preprocessing)."""
    try:
        x, fs = sf.read(str(file_path), always_2d=False)
    except Exception as exc:
        print(f"Warning: could not read {file_path}: {exc}")
        return None, None

    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x.ravel()
    if x.size < 32:
        return None, None

    if fs != WHISPER_SAMPLE_RATE:
        if librosa is None:
            raise ImportError(
                "librosa is required to resample audio to 16 kHz for Whisper. "
                "Install with: pip install librosa"
            )
        x = librosa.resample(x, orig_sr=int(fs), target_sr=WHISPER_SAMPLE_RATE)
        fs = WHISPER_SAMPLE_RATE

    return x, int(fs)


def _pool_hidden_states(hidden: torch.Tensor, pool: str) -> np.ndarray:
    """Pool (batch, time, dim) hidden states to a 1D numpy vector."""
    if pool == "mean":
        vec = hidden.mean(dim=1)
    elif pool == "mean_std":
        mean = hidden.mean(dim=1)
        std = hidden.std(dim=1, unbiased=False)
        vec = torch.cat([mean, std], dim=-1)
    else:
        raise ValueError(f"Unsupported pool mode: {pool!r}")
    return vec.squeeze(0).detach().cpu().numpy().astype(np.float32)


def extract_embedding_from_audio(
    model: torch.nn.Module,
    processor: WhisperFeatureExtractor,
    audio: np.ndarray,
    sampling_rate: int,
    *,
    device: str,
    pool: str = "mean",
) -> Optional[np.ndarray]:
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        hidden = model.encoder(input_features).last_hidden_state
    return _pool_hidden_states(hidden, pool)


def embedding_dict_from_vector(embedding: np.ndarray, prefix: str = "whisper") -> dict[str, float]:
    return {f"{prefix}_{i}": float(v) for i, v in enumerate(np.asarray(embedding, dtype=np.float32))}


def extract_file_whisper(
    file_path: str | Path,
    model: torch.nn.Module,
    processor: WhisperFeatureExtractor,
    *,
    device: str,
    pool: str = "mean",
) -> Optional[dict[str, float]]:
    audio, sr = load_audio_for_whisper(file_path)
    if audio is None or sr is None:
        return None

    embedding = extract_embedding_from_audio(
        model,
        processor,
        audio,
        sr,
        device=device,
        pool=pool,
    )
    if embedding is None:
        return None
    return embedding_dict_from_vector(embedding)


def _checkpoint(rows: list[dict], output_csv: Path) -> None:
    df = pd.DataFrame(rows)
    meta_cols = ["file_name", "speaker", "label", "task"]
    other_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + other_cols]
    df.to_csv(output_csv, index=False)


def run_extraction(
    wav_paths: list[Path],
    output_csv: Path,
    metadata_fn: Callable[[Path], dict[str, str]],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    local_model_dir: Optional[str] = None,
    device: Optional[str] = None,
    pool: str = "mean",
    save_every: int = 50,
    resume: bool = False,
) -> None:
    if not wav_paths:
        raise RuntimeError("No wav files to process")

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

    print(f"Loading Whisper from {local_model_dir or model_id} ...")
    model, processor, dev = load_whisper(
        model_id=model_id,
        device=device,
        local_model_dir=local_model_dir,
    )
    print(f"Model ready on {dev} | pool={pool} | files={len(wav_paths)}")

    for idx, wav in enumerate(wav_paths, start=1):
        if wav.stem in done:
            continue

        meta = metadata_fn(wav)
        print(
            f"[{idx}/{len(wav_paths)}] {wav} | "
            f"speaker={meta.get('speaker', '?')} label={meta.get('label', '?')} task={meta.get('task', '?')}"
        )

        feat = extract_file_whisper(wav, model, processor, device=dev, pool=pool)
        if feat is None:
            continue

        row = {
            "file_name": wav.stem,
            "speaker": meta.get("speaker", "unknown"),
            "label": meta.get("label", "unknown"),
            "task": meta.get("task", "unknown"),
            **feat,
        }
        rows.append(row)
        done.add(wav.stem)

        if save_every > 0 and len(rows) % save_every == 0:
            _checkpoint(rows, output_csv)
            print(f"Checkpoint saved: {len(rows)} rows -> {output_csv}")

    if not rows:
        raise RuntimeError("No embeddings extracted (all files failed?)")

    _checkpoint(rows, output_csv)
    print(f"Saved {len(rows)} rows -> {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Whisper encoder (outer layer) embeddings from wav files"
    )
    parser.add_argument("--wav_root", required=True, help="Root directory containing .wav files")
    parser.add_argument("--output_csv", default="features_whisper.csv", help="Output CSV path")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument(
        "--local_model_dir",
        default="",
        help="Optional local directory from huggingface-cli download",
    )
    parser.add_argument("--device", default="", help="Torch device (default: cuda if available)")
    parser.add_argument(
        "--pool",
        choices=["mean", "mean_std"],
        default="mean",
        help="Utterance pooling over Whisper encoder frames",
    )
    parser.add_argument("--max_files", type=int, default=0, help="Optional limit for quick tests")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    wav_root = Path(args.wav_root)
    wavs = sorted(wav_root.rglob("*.wav"))
    if args.max_files and args.max_files > 0:
        wavs = wavs[: args.max_files]
    if not wavs:
        raise FileNotFoundError(f"No .wav files found under: {wav_root}")

    def _default_meta(_wav: Path) -> dict[str, str]:
        return {"speaker": "unknown", "label": "unknown", "task": "unknown"}

    run_extraction(
        wavs,
        Path(args.output_csv),
        _default_meta,
        model_id=args.model_id,
        local_model_dir=args.local_model_dir or None,
        device=args.device or None,
        pool=args.pool,
        save_every=int(args.save_every),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
