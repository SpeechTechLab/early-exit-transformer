#!/usr/bin/env python3
"""Smoke-test data loading for training (CollatePaddingFn) and inference (CollateInferFn).

Reuses the same code paths as train.py / inference.py (get_data_loader, get_infer_data_loader)
and checks batch structure, channel counts, and time lengths.

Examples:
  python verify_data_loading.py --decoder_mode ctc
  python verify_data_loading.py --decoder_mode ctc --append_glottal_features --glottal_from_waveform
  python verify_data_loading.py --decoder_mode ctc --append_glottal_features \\
      --glottal_features_path data_release/small_glottal.csv --n_glottal_features 10

Defaults for a quick run (override with explicit flags):
  --train_split train-clean-100 --max_train_utts 8 --max_infer_utts 8 --n_workers 0
  --n_batch_split 1 --batch_size 4
"""

from __future__ import annotations

import os
import sys

# Same audio backend handling as data.py
os.environ.setdefault("TORCHAUDIO_USE_TORCHCODEC", "0")

from data import get_data_loader, get_infer_data_loader
from util.conf import finalize_configured_args, get_parser
from util.data_loader import infer_glottal_feature_dim


def _has_cli_flag(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(f"{flag}=") for a in argv)


def _merge_expected_input_channels(args) -> int | None:
    """Match train.py / inference.py input_features_length for waveform + optional glottal."""
    if getattr(args, "use_precomputed_features", False):
        # Precomputed tensor width varies by manifest; skip strict check unless user sets n_glottal only.
        return None
    n_mels = args.n_mels
    if not getattr(args, "append_glottal_features", False):
        return n_mels
    if getattr(args, "glottal_from_waveform", False):
        from extract_glottal_features_qcp import QCP_FRAME_FEATURE_DIM

        g = args.n_glottal_features if args.n_glottal_features > 0 else QCP_FRAME_FEATURE_DIM
        return n_mels + g
    if not args.glottal_features_path:
        raise ValueError("append_glottal_features requires glottal_features_path or glottal_from_waveform")
    if args.n_glottal_features > 0:
        return n_mels + args.n_glottal_features
    g = infer_glottal_feature_dim(
        args.glottal_features_path,
        drop_mfcc=getattr(args, "glottal_drop_mfcc", False),
    )
    return n_mels + g


def _check_training_batch(c_batch, args, expect_channels: int | None, batch_idx: int):
    """c_batch: list of (src, trg, ctc_target_len, valid_lengths) from CollatePaddingFn."""
    if not isinstance(c_batch, list):
        raise TypeError(f"Training collate must return a list, got {type(c_batch)}")
    if len(c_batch) == 0:
        raise ValueError("Training collate returned empty list")

    for chunk_i, pack in enumerate(c_batch):
        if not isinstance(pack, (list, tuple)) or len(pack) != 4:
            raise TypeError(f"Chunk {chunk_i}: expected 4-tuple, got {pack!r}")
        src, trg, ctc_target_len, valid_lengths = pack
        if src.dim() != 3:
            raise ValueError(f"Chunk {chunk_i}: src dim {src.dim()} (expected 3 [B, F, T])")
        b, f, t = src.shape
        if expect_channels is not None and f != expect_channels:
            raise AssertionError(
                f"Training batch {batch_idx} chunk {chunk_i}: expected F={expect_channels}, got {f}"
            )
        if trg.dim() != 2:
            raise ValueError(f"Chunk {chunk_i}: trg dim {trg.dim()} (expected 2)")
        if trg.size(0) != b:
            raise AssertionError(f"Chunk {chunk_i}: trg batch {trg.size(0)} != src batch {b}")
        if ctc_target_len.numel() != b:
            raise AssertionError(f"Chunk {chunk_i}: ctc_target_len length != batch size")
        if valid_lengths.numel() != b:
            raise AssertionError(f"Chunk {chunk_i}: valid_lengths length != batch size")
        # Time length consistency: each row should be <= padded T (pad_sequence uses max T in chunk).
        if (valid_lengths > t).any():
            raise AssertionError(
                f"Chunk {chunk_i}: valid_lengths exceed padded time T={t}: {valid_lengths.tolist()}"
            )


def _check_inference_batch(batch, args, expect_channels: int | None):
    if batch is None:
        raise ValueError("Inference collate returned None")
    if not isinstance(batch, (list, tuple)) or len(batch) != 3:
        raise TypeError(f"Inference collate must return 3-tuple, got {type(batch)} len={getattr(batch,'__len__',None)}")
    src, trg, t_source = batch
    if src.dim() != 3:
        raise ValueError(f"Inference src dim {src.dim()} (expected 3 [B, F, T])")
    b, f, t = src.shape
    if expect_channels is not None and f != expect_channels:
        raise AssertionError(f"Inference: expected F={expect_channels}, got {f}")
    if trg.dim() != 2 or trg.size(0) != b:
        raise AssertionError("Inference: trg batch mismatch")
    if t_source.numel() != b:
        raise AssertionError("Inference: t_source length != batch size")
    if (t_source > t).any():
        raise AssertionError(f"Inference: t_source exceeds padded T={t}: {t_source.tolist()}")


def main():
    parser = get_parser()
    parser.add_argument(
        "--verify_batches",
        type=int,
        default=2,
        help="Number of DataLoader batches to iterate per phase (train / infer).",
    )
    parser.add_argument(
        "--verify_infer_split",
        type=str,
        default="test-clean",
        help="LibriSpeech URL name for inference smoke test (e.g. test-clean, train-clean-100).",
    )
    parser.add_argument(
        "--verify_only_train",
        action="store_true",
        help="Only run training DataLoader checks.",
    )
    parser.add_argument(
        "--verify_only_infer",
        action="store_true",
        help="Only run inference DataLoader checks.",
    )
    argv = sys.argv[1:]
    if not _has_cli_flag(argv, "--decoder_mode"):
        argv = ["--decoder_mode", "ctc"] + argv
    if not _has_cli_flag(argv, "--train_split"):
        argv = ["--train_split", "train-clean-100"] + argv
    if not _has_cli_flag(argv, "--batch_size"):
        argv = ["--batch_size", "4"] + argv
    args = parser.parse_args(argv)
    args = finalize_configured_args(args)

    # Sensible smoke-test defaults (user can override on CLI).
    if args.max_train_utts <= 0:
        args.max_train_utts = 8
    if args.max_infer_utts <= 0:
        args.max_infer_utts = 8
    args.n_workers = 0
    args.n_batch_split = 1

    do_train = not args.verify_only_infer
    do_infer = not args.verify_only_train
    if args.verify_only_train and args.verify_only_infer:
        print(
            "ERROR: specify at most one of --verify_only_train / --verify_only_infer.",
            file=sys.stderr,
        )
        return 2

    expect_ch = _merge_expected_input_channels(args)
    print(
        f"verify_data_loading: expect_channels={expect_ch} "
        f"(append_glottal={getattr(args,'append_glottal_features',False)} "
        f"glottal_from_waveform={getattr(args,'glottal_from_waveform',False)})"
    )

    if do_train:
        print("--- Training DataLoader (CollatePaddingFn) ---")
        train_loader = get_data_loader(args)
        collate = train_loader.collate_fn
        n_batches = 0
        for i, c_batch in enumerate(train_loader):
            _check_training_batch(c_batch, args, expect_ch, i)
            if getattr(args, "append_glottal_features", False):
                found = getattr(collate, "glottal_found_count", 0)
                missing = getattr(collate, "glottal_missing_count", 0)
                print(f"  batch {i}: chunks={len(c_batch)} glottal_found={found} glottal_missing={missing}")
            else:
                print(f"  batch {i}: chunks={len(c_batch)}")
            n_batches += 1
            if n_batches >= args.verify_batches:
                break
        if n_batches == 0:
            raise RuntimeError("Training DataLoader produced no batches (check dataset path / filters).")
        print(f"OK: {n_batches} training batch(es) checked.")

    if do_infer:
        print("--- Inference DataLoader (CollateInferFn) ---")
        infer_loader = get_infer_data_loader(args, split=args.verify_infer_split, shuffle=False)
        collate = infer_loader.collate_fn
        n_batches = 0
        for i, batch in enumerate(infer_loader):
            _check_inference_batch(batch, args, expect_ch)
            if getattr(args, "append_glottal_features", False):
                found = getattr(collate, "glottal_found_count", 0)
                missing = getattr(collate, "glottal_missing_count", 0)
                print(f"  batch {i}: glottal_found={found} glottal_missing={missing}")
            else:
                print(f"  batch {i}: ok")
            n_batches += 1
            if n_batches >= args.verify_batches:
                break
        if n_batches == 0:
            raise RuntimeError("Inference DataLoader produced no batches.")
        print(f"OK: {n_batches} inference batch(es) checked.")

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
