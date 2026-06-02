#!/usr/bin/env python3
"""
Fine-tune Whisper on Bridge2AI using existing Zipformer manifests.

We reuse existing splits (e.g. bridge2ai_zipformer_full_all_tasks/manifests/read_{train,dev,test}.txt)
and map each manifest row (feature_csv, transcript) -> wav_path via subset_meta.tsv.

Outputs:
- checkpoints + Trainer logs under --output_dir
- eval metrics JSON via Trainer
- a small JSONL with sample predictions per eval split
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def read_manifest(manifest_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: <feature_csv_path>,<transcript>
            feat, txt = line.split(",", 1)
            rows.append((feat.strip(), txt.strip()))
    return rows


def load_feature_to_wav_map(subset_meta_tsv: Path) -> dict[str, str]:
    """
    Returns mapping: feature_csv -> wav_path
    Both paths are stored as-is from subset_meta.tsv.
    """
    out: dict[str, str] = {}
    with subset_meta_tsv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            feat = (row.get("feature_csv") or "").strip()
            wav = (row.get("wav_path") or "").strip()
            if feat and wav:
                out[feat] = wav
    return out


def build_examples(
    manifest_path: Path,
    feature_to_wav: dict[str, str],
    repo_root: Path,
    audio_root: Path,
    strict_audio: bool,
) -> list[dict[str, Any]]:
    rows = read_manifest(manifest_path)
    examples: list[dict[str, Any]] = []
    missing = 0
    missing_audio = 0
    for feat, txt in rows:
        wav_rel = feature_to_wav.get(feat)
        if wav_rel is None:
            missing += 1
            continue
        wav_path = (audio_root / wav_rel).resolve()
        if not wav_path.exists():
            missing_audio += 1
            if strict_audio:
                raise FileNotFoundError(f"Missing audio file: {wav_path}")
            continue
        examples.append(
            {
                "audio_path": str(wav_path),
                "text": txt,
                "feature_csv": feat,
            }
        )
    if missing:
        print(f"[warn] {missing} manifest rows missing from subset_meta.tsv: {manifest_path}")
    if missing_audio:
        print(
            f"[warn] {missing_audio} manifest rows missing audio under audio_root='{audio_root}': "
            f"{manifest_path} (set --audio_root or use --strict_audio)"
        )
    return examples


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # inputs: list of dicts with "input_features" + "labels"
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="openai/whisper-large-v3")
    ap.add_argument("--subset_meta_tsv", default="bridge2ai_zipformer_full_all_tasks/subset_meta.tsv")
    ap.add_argument("--train_manifest", default="bridge2ai_zipformer_full_all_tasks/manifests/read_train.txt")
    ap.add_argument("--eval_manifest", default="bridge2ai_zipformer_full_all_tasks/manifests/read_dev.txt")
    ap.add_argument("--test_manifest", default="bridge2ai_zipformer_full_all_tasks/manifests/read_test.txt")
    ap.add_argument("--output_dir", default="whisper_runs/bridge2ai_readspeech")
    ap.add_argument(
        "--audio_root",
        default=".",
        help="Base directory prepended to wav_path from subset_meta.tsv (default: repo root).",
    )
    ap.add_argument(
        "--strict_audio",
        action="store_true",
        help="Fail fast if any audio_path is missing under --audio_root (default: skip missing).",
    )
    ap.add_argument("--language", default="english")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])

    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp16", action="store_true")
    # Used both for dataset preprocessing (num_proc) and DataLoader workers.
    # Default 0 keeps things predictable in containers and avoids fork overhead.
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--save_pred_samples", type=int, default=40)
    ap.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; run Whisper baseline inference on eval/test only.",
    )
    ap.add_argument(
        "--debug_audio",
        action="store_true",
        help="Print per-utterance audio loading + feature extraction timing (first few items).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    audio_root = (repo_root / args.audio_root).resolve()
    subset_meta_tsv = (repo_root / args.subset_meta_tsv).resolve()
    feature_to_wav = load_feature_to_wav_map(subset_meta_tsv)

    # Lazy imports after args parsing (keeps error messages cleaner if deps missing)
    import torch
    import torchaudio
    import evaluate
    from datasets import Dataset
    from transformers import (
        AutoProcessor,
        AutoModelForSpeechSeq2Seq,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    import inspect

    # Build datasets from manifests.
    # In eval-only mode we intentionally skip train_manifest to avoid extra IO/warnings.
    train_ex = []
    if not args.eval_only:
        train_ex = build_examples(
            (repo_root / args.train_manifest).resolve(),
            feature_to_wav,
            repo_root,
            audio_root=audio_root,
            strict_audio=bool(args.strict_audio),
        )
    eval_ex = build_examples(
        (repo_root / args.eval_manifest).resolve(),
        feature_to_wav,
        repo_root,
        audio_root=audio_root,
        strict_audio=bool(args.strict_audio),
    )
    test_ex = build_examples(
        (repo_root / args.test_manifest).resolve(),
        feature_to_wav,
        repo_root,
        audio_root=audio_root,
        strict_audio=bool(args.strict_audio),
    )

    if args.eval_only:
        if not eval_ex and not test_ex:
            raise SystemExit(
                "No utterances with audio found for eval/test. Whisper needs wav files under "
                f"--audio_root ({audio_root}). Expected paths like "
                f"{audio_root}/bridge2ai_adult_wav_v2/<id>__<session>__<task>.wav "
                "(see subset_meta.tsv). Copy bridge2ai_adult_wav_v2/ to the repo or set --audio_root."
            )
        if train_ex:
            print(f"[info] eval_only: ignoring {len(train_ex)} train rows with audio")
    else:
        if not train_ex:
            raise SystemExit(
                f"No training utterances with audio under --audio_root ({audio_root}). "
                "All manifest rows were skipped (missing wav). Copy bridge2ai_adult_wav_v2/ into the "
                "repo root or pass --audio_root to the directory that contains it. "
                "Generate a transfer list with: python3 export_whisper_wav_paths.py --manifest <path>"
            )

    ds_train = Dataset.from_list(train_ex if train_ex else eval_ex[:1])
    ds_eval = Dataset.from_list(eval_ex)
    ds_test = Dataset.from_list(test_ex)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[info] torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device}")

    # Force language/task tokens (stabilizes decoding for English-only ASR).
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "set_prefix_tokens"):
        processor.tokenizer.set_prefix_tokens(language=args.language, task=args.task)

    target_sr = processor.feature_extractor.sampling_rate

    def _load_and_resample(path: str) -> np.ndarray:
        # torchaudio.load may hang in minimal containers when it falls back to FFmpeg.
        # Use soundfile for robust wav decoding; keep torchaudio only for resampling.
        try:
            import soundfile as sf

            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            if audio is None:
                raise RuntimeError("soundfile returned None")
            # mixdown to mono if needed
            if getattr(audio, "ndim", 1) == 2:
                audio = audio.mean(axis=1).astype(np.float32, copy=False)
            audio_t = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        except Exception as exc:
            print(f"[warn] soundfile decode failed for {path!r}: {exc}; falling back to torchaudio.load")
            wav, sr = torchaudio.load(path)
            if wav.ndim == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            audio_t = wav.squeeze(0).float()

        if int(sr) != int(target_sr):
            audio_t = torchaudio.functional.resample(audio_t, int(sr), int(target_sr))
        return audio_t.cpu().numpy()

    def prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
        import time

        p = batch["audio_path"]
        t0 = time.time()
        audio = _load_and_resample(p)
        t1 = time.time()
        feats = processor.feature_extractor(audio, sampling_rate=target_sr).input_features[0]
        t2 = time.time()
        if args.debug_audio:
            # This runs inside datasets.map; keep output short and informative.
            dur = len(audio) / float(target_sr) if target_sr else 0.0
            print(
                f"[debug_audio] {Path(p).name} | {dur:.1f}s | "
                f"decode={t1 - t0:.2f}s feat={t2 - t1:.2f}s",
                flush=True,
            )
        batch["input_features"] = feats
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    # Map in-process (small-ish datasets); for bigger runs, switch to batched map + caching.
    print("[info] preprocessing audio -> whisper log-mels (CPU). This can take a while...")
    num_proc = max(int(args.num_workers), 1)
    if not args.eval_only:
        ds_train = ds_train.map(prepare_batch, remove_columns=ds_train.column_names, num_proc=num_proc)
    if len(eval_ex):
        ds_eval = ds_eval.map(prepare_batch, remove_columns=ds_eval.column_names, num_proc=num_proc)
    if len(test_ex):
        ds_test = ds_test.map(prepare_batch, remove_columns=ds_test.column_names, num_proc=num_proc)
    print("[info] preprocessing done.")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": float(wer)}

    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transformers renamed some TrainingArguments fields across versions.
    # Keep this script runnable in older/newer container images.
    ta_sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    max_steps = 0 if args.eval_only else args.max_steps
    ta_kwargs: dict[str, Any] = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=max_steps,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
    )
    if "evaluation_strategy" in ta_sig.parameters:
        ta_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_sig.parameters:
        ta_kwargs["eval_strategy"] = "steps"
    else:
        raise TypeError(
            "Unsupported transformers version: Seq2SeqTrainingArguments has neither "
            "'evaluation_strategy' nor 'eval_strategy'."
        )
    training_args = Seq2SeqTrainingArguments(**ta_kwargs)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    if not args.eval_only:
        print("[info] starting training (GPU if available).")
        trainer.train()
    else:
        print("[info] eval_only: skipping training; running generate on eval/test.")

    # Evaluate on dev + test and dump a small sample of predictions.
    def _predict_and_dump(split_name: str, ds) -> dict[str, Any]:
        if ds is None or len(ds) == 0:
            print(f"[warn] skip predict: empty {split_name} split")
            return {}
        pred = trainer.predict(ds, max_length=225)
        metrics = {f"{split_name}_{k}": float(v) for k, v in pred.metrics.items()}

        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        n = min(int(args.save_pred_samples), len(pred_str))
        out_jsonl = out_dir / f"pred_samples_{split_name}.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps({"ref": label_str[i], "hyp": pred_str[i]}, ensure_ascii=False) + "\n")
        return metrics

    all_metrics: dict[str, Any] = {}
    if len(eval_ex):
        all_metrics.update(_predict_and_dump("dev", ds_eval))
    if len(test_ex):
        all_metrics.update(_predict_and_dump("test", ds_test))

    (out_dir / "final_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print("Wrote:", out_dir / "final_metrics.json")


if __name__ == "__main__":
    # Keep tokenizers from spawning many threads inside workers.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

