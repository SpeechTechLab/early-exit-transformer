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

    # Build datasets from manifests.
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

    ds_train = Dataset.from_list(train_ex)
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
        wav, sr = torchaudio.load(path)
        # mixdown to mono
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.cpu().numpy()

    def prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
        audio = _load_and_resample(batch["audio_path"])
        batch["input_features"] = processor.feature_extractor(audio, sampling_rate=target_sr).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    # Map in-process (small-ish datasets); for bigger runs, switch to batched map + caching.
    print("[info] preprocessing audio -> whisper log-mels (CPU). This can take a while...")
    num_proc = max(int(args.num_workers), 1)
    ds_train = ds_train.map(prepare_batch, remove_columns=ds_train.column_names, num_proc=num_proc)
    ds_eval = ds_eval.map(prepare_batch, remove_columns=ds_eval.column_names, num_proc=num_proc)
    ds_test = ds_test.map(prepare_batch, remove_columns=ds_test.column_names, num_proc=num_proc)
    print("[info] preprocessing done; starting training (GPU if available).")

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

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
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

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on dev + test and dump a small sample of predictions.
    def _predict_and_dump(split_name: str, ds) -> dict[str, Any]:
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
    all_metrics.update(_predict_and_dump("dev", ds_eval))
    all_metrics.update(_predict_and_dump("test", ds_test))

    (out_dir / "final_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print("Wrote:", out_dir / "final_metrics.json")


if __name__ == "__main__":
    # Keep tokenizers from spawning many threads inside workers.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

