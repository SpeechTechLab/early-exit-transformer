#!/usr/bin/env python3
"""Export English test references and (optional) Whisper CE/CTC hypotheses.

Datasets (test only — not dev):
  CE  : clean_short_en/read_test.txt  (239 utts) — cross-entropy seq2seq fine-tune
  CTC : subset_meta split=test        (6100 utts) — Bridge2AI NORM CTC fine-tune

References are written as TSV + JSONL under classification_results/whisper_en_asr_test/.

Hypotheses require checkpoints on disk:
  CE  : whisper_runs/bridge2ai_read_clean_short_en_ft_v2/checkpoint-150
  CTC : whisper_ctc_runs/bridge2ai_norm_partial_unfreeze/whisper_ctc_partial_unfreeze.pth
        (+ SLAM-LLM inference.py on the cluster)

Examples:
  # References only (no GPU / no checkpoint)
  python3 scripts/export_whisper_en_test_asr.py --refs-only

  # CE test hypotheses (needs checkpoint + wavs)
  python3 scripts/export_whisper_en_test_asr.py \\
    --ce-checkpoint whisper_runs/bridge2ai_read_clean_short_en_ft_v2/checkpoint-150

  # Both references + CE hyps
  python3 scripts/export_whisper_en_test_asr.py --refs-only --ce-checkpoint ...
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "classification_results" / "whisper_en_asr_test"

CE_TEST_MANIFEST = REPO_ROOT / "bridge2ai_zipformer_full_all_tasks/manifests/clean_short_en/read_test.txt"
SUBSET_META = REPO_ROOT / "bridge2ai_zipformer_full_all_tasks/subset_meta.tsv"
CTC_TEST_JSONL = REPO_ROOT / "bridge2ai_norm_ctc/jsonl/test.jsonl"


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


def export_ce_references(feature_to_wav: dict[str, str]) -> list[dict]:
    records: list[dict] = []
    missing = 0
    for feat, ref in read_manifest(CE_TEST_MANIFEST):
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
                "split": "test",
                "dataset": "clean_short_en",
                "objective": "cross_entropy",
            }
        )
    if missing:
        print(f"[warn] CE test: {missing} rows missing from subset_meta", flush=True)
    return records


def export_ctc_references(feature_to_wav: dict[str, str]) -> list[dict]:
    if CTC_TEST_JSONL.is_file():
        records: list[dict] = []
        with CTC_TEST_JSONL.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                wav = Path(rec["source"])
                records.append(
                    {
                        "utt_id": wav.stem,
                        "feature_csv": "",
                        "wav_path": str(wav.resolve()),
                        "reference": normalize_text(rec["target"]),
                        "split": "test",
                        "dataset": "bridge2ai_norm",
                        "objective": "ctc",
                    }
                )
        return records

    records = []
    missing_wav = 0
    with SUBSET_META.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if (row.get("split") or "").strip() != "test":
                continue
            feat = (row.get("feature_csv") or "").strip()
            wav_rel = (row.get("wav_path") or "").strip()
            transcript = (row.get("transcript") or "").strip()
            if not feat or not wav_rel or not transcript:
                continue
            wav_path = (REPO_ROOT / wav_rel).resolve()
            if not wav_path.is_file():
                missing_wav += 1
                continue
            records.append(
                {
                    "utt_id": feature_to_utt_id(feat),
                    "feature_csv": feat,
                    "wav_path": str(wav_path),
                    "reference": normalize_text(transcript),
                    "split": "test",
                    "dataset": "bridge2ai_norm",
                    "objective": "ctc",
                }
            )
    if missing_wav:
        print(f"[warn] CTC test: {missing_wav} rows missing wav files", flush=True)
    return records


def write_split(records: list[dict], stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tsv_path = OUT_DIR / f"{stem}.tsv"
    jsonl_path = OUT_DIR / f"{stem}.jsonl"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["utt_id", "feature_csv", "wav_path", "reference", "hypothesis", "split", "dataset", "objective"],
            delimiter="\t",
            extrasaction="ignore",
        )
        w.writeheader()
        for r in records:
            row = dict(r)
            row.setdefault("hypothesis", "")
            w.writerow(row)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({**r, "hypothesis": r.get("hypothesis", "")}, ensure_ascii=False) + "\n")
    print(f"Wrote {tsv_path} ({len(records)} utts)")
    print(f"Wrote {jsonl_path}")


def run_ce_hypotheses(checkpoint: Path, records: list[dict], batch_size: int) -> list[dict]:
    """Decode CE checkpoint on CE test wavs; fill hypothesis field."""
    try:
        import numpy as np
        import torch
        import torchaudio
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as exc:
        raise SystemExit(f"CE decode needs torch/transformers/torchaudio: {exc}") from exc

    ckpt = checkpoint.resolve()
    if not ckpt.is_dir():
        raise SystemExit(f"CE checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(str(ckpt))
    model = WhisperForConditionalGeneration.from_pretrained(str(ckpt)).to(device)
    model.eval()
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
            raise SystemExit(f"Missing wav(s), e.g. {missing[0]}")

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
        print(f"  CE decode {min(i + batch_size, len(records))}/{len(records)}", flush=True)
    return out


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    try:
        import evaluate
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "evaluate"], check=True)
        import evaluate
    wer = evaluate.load("wer")
    return float(wer.compute(predictions=hyps, references=refs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refs-only", action="store_true", help="Export test references only")
    parser.add_argument(
        "--ce-checkpoint",
        type=Path,
        default=None,
        help="Finetuned CE checkpoint dir (e.g. .../checkpoint-150)",
    )
    parser.add_argument("--ce-batch-size", type=int, default=4)
    parser.add_argument(
        "--ctc-checkpoint",
        type=Path,
        default=None,
        help="CTC .pth checkpoint (decode via SLAM-LLM on cluster; not implemented here)",
    )
    args = parser.parse_args()

    feature_to_wav = load_feature_to_wav()
    ce_refs = export_ce_references(feature_to_wav)
    ctc_refs = export_ctc_references(feature_to_wav)

    write_split(ce_refs, "ce_test_references")
    write_split(ctc_refs, "ctc_test_references")

    if args.refs_only and not args.ce_checkpoint:
        print("\nReferences exported. For hypotheses, re-run with --ce-checkpoint on a machine with wavs + GPU.")
        print("CTC decode: use SLAM-LLM inference.py on the cluster (see bridge2ai_norm_ctc/README.md).")
        return 0

    if args.ce_checkpoint:
        print("\nDecoding CE checkpoint on clean_short_en test ...", flush=True)
        ce_out = run_ce_hypotheses(args.ce_checkpoint, ce_refs, args.ce_batch_size)
        write_split(ce_out, "ce_test_ref_hyp")
        wer = compute_wer([r["reference"] for r in ce_out], [r["hypothesis"] for r in ce_out])
        metrics = {
            "objective": "cross_entropy",
            "checkpoint": str(args.ce_checkpoint),
            "dataset": "clean_short_en",
            "split": "test",
            "n_utts": len(ce_out),
            "wer": wer,
            "wer_percent": round(100 * wer, 2),
        }
        metrics_path = OUT_DIR / "ce_test_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"CE test WER: {metrics['wer_percent']}%  ->  {metrics_path}")

    if args.ctc_checkpoint:
        print(
            "\nCTC full-test decode is not run in this script (needs SLAM-LLM on cluster).\n"
            "After inference, merge hypotheses into ctc_test_references.tsv with hypothesis column.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
