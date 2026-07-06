#!/usr/bin/env python3
"""Fair matched ASR eval: CE and CTC Whisper on the same English test set.

Uses clean_short_en/read_test.txt (239 utts) for all models.

Outputs (under classification_results/whisper_en_asr_test/):
  clean_short_en_test.jsonl          — wav + reference
  baseline_matched_test_ref_hyp.tsv  — non-fine-tuned Whisper (optional)
  ce_matched_test_ref_hyp.tsv        — CE fine-tuned hypotheses
  ctc_matched_test_ref_hyp.tsv       — CTC fine-tuned hypotheses
  matched_asr_comparison.json        — WER summary

Example (cluster):
  python3 scripts/run_matched_whisper_asr_eval.py --baseline \\
    --ce-checkpoint whisper_runs/bridge2ai_read_clean_short_en_ft_v2/checkpoint-150 \\
    --ctc-checkpoint whisper_ctc_runs/bridge2ai_norm_partial_unfreeze/whisper_ctc_partial_unfreeze.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "classification_results" / "whisper_en_asr_test"
TEST_MANIFEST = REPO_ROOT / "bridge2ai_zipformer_full_all_tasks/manifests/clean_short_en/read_test.txt"
SUBSET_META = REPO_ROOT / "bridge2ai_zipformer_full_all_tasks/subset_meta.tsv"
DEFAULT_MODEL_ID = "openai/whisper-large-v3"
DEFAULT_CE_CKPT = REPO_ROOT / "whisper_runs/bridge2ai_read_clean_short_en_ft_v2/checkpoint-150"
DEFAULT_CTC_CKPT = (
    REPO_ROOT / "whisper_ctc_runs/bridge2ai_norm_partial_unfreeze/whisper_ctc_partial_unfreeze.pth"
)


def _whisper_processor_source(checkpoint: Path, base_model_id: str) -> str:
    """HF Trainer checkpoints may omit tokenizer files; fall back to base model."""
    has_preproc = (checkpoint / "preprocessor_config.json").is_file()
    has_tok = (checkpoint / "tokenizer.json").is_file() or (checkpoint / "vocab.json").is_file()
    return str(checkpoint) if has_preproc and has_tok else base_model_id


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def feature_to_utt_id(feature_csv: str) -> str:
    return Path(feature_csv).stem.replace(" ", "_")


def load_feature_to_wav() -> dict[str, str]:
    out: dict[str, str] = {}
    with SUBSET_META.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            feat = (row.get("feature_csv") or "").strip()
            wav = (row.get("wav_path") or "").strip()
            if feat and wav:
                out[feat] = wav
    return out


def read_manifest(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            feat, txt = line.split(",", 1)
            rows.append((feat.strip(), txt.strip()))
    return rows


def build_test_records(feature_to_wav: dict[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing = 0
    for feat, ref in read_manifest(TEST_MANIFEST):
        wav_rel = feature_to_wav.get(feat)
        if not wav_rel:
            missing += 1
            continue
        wav_path = (REPO_ROOT / wav_rel).resolve()
        records.append(
            {
                "utt_id": feature_to_utt_id(feat),
                "feature_csv": feat,
                "wav_path": str(wav_path),
                "reference": normalize_text(ref),
            }
        )
    if missing:
        print(f"[warn] {missing} test rows missing from subset_meta", flush=True)
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps({"source": r["wav_path"], "target": r["reference"]}, ensure_ascii=False)
                + "\n"
            )


def write_ref_hyp_tsv(records: list[dict], path: Path, objective: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["utt_id", "wav_path", "reference", "hypothesis", "objective", "dataset"],
            delimiter="\t",
        )
        w.writeheader()
        for r in records:
            w.writerow(
                {
                    "utt_id": r["utt_id"],
                    "wav_path": r["wav_path"],
                    "reference": r["reference"],
                    "hypothesis": r.get("hypothesis", ""),
                    "objective": objective,
                    "dataset": "clean_short_en_test",
                }
            )


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    import evaluate

    wer = evaluate.load("wer")
    return float(wer.compute(predictions=hyps, references=refs))


def run_seq2seq_decode(
    records: list[dict],
    batch_size: int,
    *,
    base_model_id: str = DEFAULT_MODEL_ID,
    checkpoint: Path | None = None,
    progress_label: str = "decode",
) -> list[dict]:
    """Seq2seq Whisper decode (baseline or CE-finetuned checkpoint)."""
    import torch
    import torchaudio
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint is None:
        print(f"[info] loading base model {base_model_id}", flush=True)
        processor = WhisperProcessor.from_pretrained(base_model_id)
        model = WhisperForConditionalGeneration.from_pretrained(base_model_id).to(device).eval()
    else:
        ckpt = checkpoint.resolve()
        if not ckpt.is_dir():
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        proc_src = _whisper_processor_source(ckpt, base_model_id)
        if proc_src != str(ckpt):
            print(f"[info] processor from {proc_src} (checkpoint has no tokenizer)", flush=True)
        processor = WhisperProcessor.from_pretrained(proc_src)
        model = WhisperForConditionalGeneration.from_pretrained(str(ckpt)).to(device).eval()

    try:
        model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="en", task="transcribe"
        )
    except Exception:
        pass

    out: list[dict] = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        paths = [r["wav_path"] for r in batch]
        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            raise SystemExit(f"Missing wav: {missing[0]}")

        mels = []
        for p in paths:
            wav, sr = torchaudio.load(p)
            if wav.ndim == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0).float()
            if int(sr) != 16000:
                wav = torchaudio.functional.resample(wav, int(sr), 16000)
            feats = processor.feature_extractor(wav.numpy(), sampling_rate=16000).input_features[0]
            mels.append(feats)

        inputs = processor.feature_extractor.pad(
            [{"input_features": m} for m in mels], return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            pred_ids = model.generate(inputs.input_features, max_length=225)
        hyps = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        for r, hyp in zip(batch, hyps):
            out.append({**r, "hypothesis": normalize_text(hyp)})
        print(f"  {progress_label} {min(i + batch_size, len(records))}/{len(records)}", flush=True)
    return out


def run_ce_decode(
    records: list[dict],
    checkpoint: Path,
    batch_size: int,
    *,
    base_model_id: str = DEFAULT_MODEL_ID,
) -> list[dict]:
    return run_seq2seq_decode(
        records,
        batch_size,
        base_model_id=base_model_id,
        checkpoint=checkpoint,
        progress_label="CE",
    )


def _setup_slam_llm(slam_root: Path) -> None:
    src = slam_root / "src"
    for p in (str(src), str(slam_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


def run_ctc_decode(
    records: list[dict],
    checkpoint: Path,
    jsonl_path: Path,
    batch_size: int,
    slam_root: Path,
) -> list[dict]:
    if not slam_root.is_dir():
        raise SystemExit(
            f"SLAM-LLM not found at {slam_root}. Set SLAM_LLM_ROOT or pass --slam-llm-root."
        )
    _setup_slam_llm(slam_root)

    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from whisper_ctc.train_partial_unfreeze import (  # type: ignore
        BLANK_ID,
        CTCDataCollator,
        LocalJsonlDataset,
        WhisperCTCModel,
        id_to_char,
    )

    ckpt = checkpoint.resolve()
    if not ckpt.is_file():
        raise SystemExit(f"CTC checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = OmegaConf.create(
        {
            "encoder_path": "large-v3-turbo",
            "encoder_dim": 1280,
            "encoder_name": "whisper",
            "whisper_decode": False,
            "encoder_path_hf": None,
        }
    )
    model = WhisperCTCModel(config)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    dataset = LocalJsonlDataset(str(jsonl_path))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=CTCDataCollator(model=model),
    )

    out: list[dict] = []
    idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="CTC decode"):
            input_features = batch["input_features"].to(device)
            mel_lens = batch["mel_lens"].to(device)
            logits = model(input_features, mel_lens=mel_lens)["logits"]
            batch_size_actual = logits.shape[0]
            out_lens = (mel_lens + 1) // 2
            pred_ids = torch.argmax(logits, dim=-1)
            for b in range(batch_size_actual):
                res, prev = [], None
                for i in range(out_lens[b]):
                    val = pred_ids[b, i].item()
                    if val != prev and val != BLANK_ID:
                        res.append(id_to_char.get(val, ""))
                    prev = val
                hyp = normalize_text("".join(res).replace("|", " ").strip())
                out.append({**records[idx], "hypothesis": hyp})
                idx += 1
    if idx != len(records):
        raise RuntimeError(f"CTC decode count mismatch: {idx} vs {len(records)}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ce-checkpoint", type=Path, default=None)
    parser.add_argument("--ctc-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also decode non-fine-tuned Whisper (--model-id, default openai/whisper-large-v3)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Decode baseline only (skip CE/CTC unless checkpoints passed explicitly)",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Base Whisper id for processor fallback")
    parser.add_argument("--ce-batch-size", type=int, default=4)
    parser.add_argument("--ctc-batch-size", type=int, default=8)
    parser.add_argument(
        "--slam-llm-root",
        type=Path,
        default=Path(os.environ.get("SLAM_LLM_ROOT", "/stek/patsoura/SLAM-LLM-main")),
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only write clean_short_en_test.jsonl")
    args = parser.parse_args()

    feature_to_wav = load_feature_to_wav()
    records = build_test_records(feature_to_wav)
    if not records:
        raise SystemExit("No test records — check manifest and wav paths.")

    jsonl_path = OUT_DIR / "clean_short_en_test.jsonl"
    write_jsonl(records, jsonl_path)
    print(f"Wrote {jsonl_path} ({len(records)} utts)")

    if args.prepare_only:
        return 0

    ce_ckpt = args.ce_checkpoint or DEFAULT_CE_CKPT
    ctc_ckpt = args.ctc_checkpoint or DEFAULT_CTC_CKPT
    run_baseline = args.baseline or args.baseline_only
    run_ce = (not args.baseline_only) and (args.ce_checkpoint is not None or ce_ckpt.is_dir())
    run_ctc = (not args.baseline_only) and (args.ctc_checkpoint is not None or ctc_ckpt.is_file())

    if not run_baseline and not run_ce and not run_ctc:
        print(
            "\nNothing to decode. Use --baseline, --baseline-only, or pass checkpoints:\n"
            "  --ce-checkpoint whisper_runs/.../checkpoint-150\n"
            "  --ctc-checkpoint whisper_ctc_runs/.../whisper_ctc_partial_unfreeze.pth"
        )
        return 1

    comparison: dict[str, Any] = {
        "dataset": "clean_short_en/read_test.txt",
        "n_utts": len(records),
        "jsonl": str(jsonl_path),
    }

    if run_baseline:
        print(f"\nBaseline decode: {args.model_id}", flush=True)
        base_out = run_seq2seq_decode(
            records,
            args.ce_batch_size,
            base_model_id=args.model_id,
            checkpoint=None,
            progress_label="baseline",
        )
        base_tsv = OUT_DIR / "baseline_matched_test_ref_hyp.tsv"
        write_ref_hyp_tsv(base_out, base_tsv, "baseline")
        base_wer = compute_wer([r["reference"] for r in base_out], [r["hypothesis"] for r in base_out])
        comparison["baseline"] = {
            "model_id": args.model_id,
            "wer": base_wer,
            "wer_percent": round(100 * base_wer, 2),
            "output_tsv": str(base_tsv),
        }
        print(f"Baseline test WER: {comparison['baseline']['wer_percent']}%")

    if run_ce:
        print(f"\nCE decode: {ce_ckpt}", flush=True)
        ce_out = run_ce_decode(records, ce_ckpt, args.ce_batch_size, base_model_id=args.model_id)
        ce_tsv = OUT_DIR / "ce_matched_test_ref_hyp.tsv"
        write_ref_hyp_tsv(ce_out, ce_tsv, "cross_entropy")
        ce_wer = compute_wer([r["reference"] for r in ce_out], [r["hypothesis"] for r in ce_out])
        comparison["cross_entropy"] = {
            "checkpoint": str(ce_ckpt),
            "wer": ce_wer,
            "wer_percent": round(100 * ce_wer, 2),
            "output_tsv": str(ce_tsv),
        }
        print(f"CE test WER: {comparison['cross_entropy']['wer_percent']}%")

    if run_ctc:
        print(f"\nCTC decode: {ctc_ckpt}", flush=True)
        ctc_out = run_ctc_decode(
            records, ctc_ckpt, jsonl_path, args.ctc_batch_size, args.slam_llm_root
        )
        ctc_tsv = OUT_DIR / "ctc_matched_test_ref_hyp.tsv"
        write_ref_hyp_tsv(ctc_out, ctc_tsv, "ctc")
        ctc_wer = compute_wer([r["reference"] for r in ctc_out], [r["hypothesis"] for r in ctc_out])
        comparison["ctc"] = {
            "checkpoint": str(ctc_ckpt),
            "wer": ctc_wer,
            "wer_percent": round(100 * ctc_wer, 2),
            "output_tsv": str(ctc_tsv),
        }
        print(f"CTC test WER: {comparison['ctc']['wer_percent']}%")

    metrics_path = OUT_DIR / "matched_asr_comparison.json"
    metrics_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nWrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
