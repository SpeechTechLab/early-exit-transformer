# Bridge2AI read-speech ASR experiments (Jun 2026)

Report of ASR work on **Bridge2AI read speech** using the local **early-exit Zipformer** (`early_zipformer_2layer_exits`) and **Whisper large-v3**, with **LibriSpeech 100h pretraining** as the acoustic-model init for Zipformer.

**Repo:** [SpeechTechLab/early-exit-transformer](https://github.com/SpeechTechLab/early-exit-transformer)  
**Branch:** `irene/ee-ipatsoura-stek-1`  
**Cluster workdir:** `/stek/patsoura/early-exit-transformer` (Docker on FBK STEK)

**Note:** There is **no HuBERT fine-tuning** in this project. HuBERT-related scripts (`extract_hubert_embeddings.py`, etc.) are for **embedding extraction only**.

---

## 1. LibriSpeech pretraining (baseline init)

All Bridge2AI Zipformer runs start from a checkpoint trained on **LibriSpeech train-clean-100** (~100 h), **50 epochs**, **mel-only** (80-dim log-mels as precomputed CSV features).

| Setting | Value |
|--------|--------|
| Model | `early_zipformer_2layer_exits` |
| Exits / layers | 6 exits × 2 Conformer layers per exit |
| Decoder | CTC + LibriSpeech SentencePiece BPE (256) |
| Init checkpoint | `trained_model_zip2layer_100h_mels_only_50ep/mod049-transformer` |

### LibriSpeech evaluation (Exit 6 WER)

| Split | WER |
|-------|-----|
| **test-clean** (2620 utts) | **15.88%** |
| **test-other** (2939 utts) | **42.44%** |

(Glottal-augmented Libri training was also run; mels-only was slightly better. See `experiment_summary_20260507.md`.)

---

## 2. Bridge2AI data and manifests

Read-speech subset under `bridge2ai_zipformer_full_all_tasks/`:

| Manifest | Train | Dev | Test | Notes |
|----------|------:|----:|-----:|-------|
| `manifests/read_{train,dev,test}.txt` | 1370 | 254 | 371 | Cleaned (dedupe, drop short refs, exclude passage-8/9) |
| `manifests/clean_short/` | 925 | 158 | 239 | Also excludes all `*passage*` tasks (avoids mel truncation mismatch) |
| `manifests/clean_short_en/` | 925 | 157 | 239 | ASCII-only transcripts (English-focused eval) |

- **Features for Zipformer:** precomputed mel CSVs in `mels/` (~2.8 GB for read manifests; not in git).
- **Audio for Whisper:** `bridge2ai_adult_wav_v2/*.wav` mapped via `subset_meta.tsv` (~1321 wavs for `clean_short_en`; transferred separately to cluster).

Cleaning script: `clean_bridge2ai_manifest.py`.

---

## 3. Zipformer experiments on Bridge2AI

Training: `train.py --use_precomputed_features --n_glottal_features 80` (mel CSVs only; flag name is legacy).  
Inference: `inference.py`, report **Exit 6** WER unless noted.

### 3.1 Libri-only (no Bridge2AI finetune)

| Checkpoint | Dev manifest | Exit 6 WER | Notes |
|------------|--------------|------------|-------|
| `mod049` (Libri 100h) | `read_dev.txt` (254 utts) | **~97%** | No domain adaptation; model fails on pathological read speech |
| `mod049` | `clean_short_en/read_dev.txt` (157 utts) | **79.45%** | Same Libri model, shorter English-only dev |

### 3.2 Bridge2AI finetune from Libri init (earlier CPU / local pilots)

| Run | Train data | Best checkpoint | Dev WER (exit 6) | Test WER |
|-----|------------|-----------------|------------------|----------|
| Read-speech finetune (larger manifest, ~May 26) | ~2906 utts manifest | `mod016` | **~59%** | **~57%** |
| Same finetune re-eval on **current** cleaned `read_dev.txt` | — | `mod016` | **~81%** | — |
| Training from scratch | Bridge2AI only | — | **~97%** | Collapsed outputs |

Early stopping around epoch 16–24; best train loss ~3.44 at `mod016`.

### 3.3 GPU finetune on cluster (Jun 2, full `read_train.txt`)

| Setting | Value |
|--------|--------|
| Init | `mod049-transformer` |
| Manifest | `manifests/read_train.txt` (1370 utts) |
| LR / epochs | `init_lr=3e-6`, 80 epochs |
| Output | `trained_model_gpu_read/mod079-transformer` (last epoch saved) |
| Log | `logs/train_gpu_read.log` |

| Checkpoint | Dev (`read_dev.txt`, 254 utts) Exit 6 WER |
|------------|---------------------------------------------|
| `mod079` (epoch 79) | **68.99%** |

Improvement over Libri-only (~97% → ~69%), but still poor for production STT.

### 3.4 Manifest cleaning ablations (GPU, Jun 2)

Goal: remove long **passage** utterances that were truncated during training (mel frames clipped to 4001 while refs stayed full-length) and reduce label noise.

| Train manifest | Eval manifest | Checkpoint | Exit 6 dev WER |
|----------------|---------------|------------|----------------|
| `clean_short/read_train.txt` | `clean_short/read_dev.txt` (158) | `mod039` | **78.46%** |
| `clean_short_en/read_train.txt` | `clean_short_en/read_dev.txt` (157) | `mod039` | **80.34%** |

On the **same** `clean_short_en` dev, Libri-finetuned `mod079` reached **79.45%** — only ~1 point worse than the clean-short specialist run. **Data cleaning alone did not fix Zipformer**; decoding/domain mismatch and small-data limits dominate (~78–80% WER plateau).

---

## 4. Whisper large-v3 experiments (Jun 2)

Script: `whisper_bridge2ai_finetune.py`  
Model: `openai/whisper-large-v3`  
Eval split: `clean_short_en` (157 dev / 239 test) unless noted.

Infrastructure fixes applied during the run:

- Force **English transcribe** (`language=en`, `task=transcribe`) to avoid Spanish/Cyrillic outputs.
- Decode wavs with **soundfile** (avoid `torchaudio`/FFmpeg hang in Docker).
- Preprocess with **`num_workers=0`** (avoid `datasets.map` subprocess hang with CUDA).

### 4.1 Baseline Whisper (no finetune)

| Run | Dev WER | Test WER | Output dir |
|-----|---------|----------|------------|
| First baseline (no language forcing) | 51.96% | 33.63% | `whisper_runs/bridge2ai_read_clean_short_en_baseline/` |
| Baseline with forced English | **37.44%** | **32.62%** | `whisper_runs/bridge2ai_read_clean_short_en_baseline_enforced/` |

Sample outputs look reasonable (English hypotheses; occasional word swaps vs Harvard-style refs). A few Spanish refs still pass the ASCII filter and hurt WER when Whisper translates them to English.

### 4.2 Whisper finetune (1000 steps, Jun 2)

| Setting | Value |
|--------|--------|
| Train | `clean_short_en/read_train.txt` (925 utts) |
| Steps | 1000 (`eval_steps=200`, `save_total_limit=3`) |
| Output | `whisper_runs/bridge2ai_read_clean_short_en_ft/` |

| Checkpoint | Dev WER (during training) | Kept on disk? |
|------------|---------------------------|---------------|
| Step 200 | **22.19%** | No (deleted) |
| Step 400 | 28.96% | No |
| Step 600 | 53.51% | Yes |
| Step 800 | 46.17% | Yes |
| Step 1000 (final) | 40.13% | Yes |

**Final eval (step 1000 model):** dev **40.13%**, test **38.74%** — **worse than baseline** (overfitting; train loss → ~0).

Only checkpoints **600, 800, 1000** remain; all are worse than the **37.4%** baseline. The promising **~22%** dev at step 200 was not retained.

**Why step 200 was lost:** `Seq2SeqTrainingArguments` had `save_total_limit=3`, so HuggingFace Trainer deleted older checkpoints whenever a new one was saved (200 → 400 → 600 … left only the last three).

### 4.3 Whisper finetune v2 (400 steps, Jun 3) — **recommended model**

Re-run with `save_total_limit=15`, `load_best_model_at_end=True` (script default), and eval/save every 50 steps.

| Setting | Value |
|--------|--------|
| Train / eval / test | `clean_short_en/read_{train,dev,test}.txt` (925 / 157 / 239 utts) |
| Steps | 400 (`eval_steps=50`, `save_steps=50`) |
| LR | `1e-5`, batch 1 × grad accum 8, fp16 |
| Output | `whisper_runs/bridge2ai_read_clean_short_en_ft_v2/` |
| Metrics file | `whisper_runs/bridge2ai_read_clean_short_en_ft_v2/final_metrics.json` |

**Dev WER during training (selected steps):**

| Step | Dev WER | Notes |
|------|---------|-------|
| 50 | 14.4% | |
| 100 | 16.2% | |
| **150** | **14.0%** | **Best checkpoint (`checkpoint-150`)** |
| 200 | 17.1% | |
| 300 | 72.6% | Overfit / unstable eval |
| 400 | 36.4% | Last step (not used) |

**Final eval** (`load_best_model_at_end` → `checkpoint-150`):

| Split | WER |
|-------|-----|
| **Dev** (157 utts) | **14.03%** |
| **Test** (239 utts) | **8.40%** |

Improvement over forced-English baseline: **−23.4 pp dev**, **−24.2 pp test**.

Sample quality is good on test (e.g. Harvard sentences transcribed correctly; minor spelling variants like grey/gray). Occasional **Spanish hypotheses on English refs** still appear on dev (e.g. “today the chicken is a common dish” → Spanish output) — keep `language=en` at decode time and consider tightening the English-only manifest filter.

**Production checkpoint:** `whisper_runs/bridge2ai_read_clean_short_en_ft_v2/checkpoint-150` (~6 GB). Checkpoints 50–400 all retained on disk; delete non-best checkpoints to save ~126 GB if needed.

---

## 5. Summary comparison (Exit 6 / word-level WER)

Best results on **`clean_short_en` dev (157 utts)** where both systems were evaluated:

| System | Dev WER | Test WER | Use for STT? |
|--------|---------|----------|--------------|
| Zipformer Libri `mod049` | 79.5% | — | No |
| Zipformer Bridge2AI `mod079` | ~69% on full dev; ~78–80% on clean_short_en | — | No |
| Zipformer Bridge2AI `mod016` (best Zipformer finetune) | ~59% (older manifest) | ~57% | No |
| Whisper baseline (en forced) | 37.4% | 32.6% | Yes (fallback) |
| Whisper finetune v1 final (step 1000) | 40.1% | 38.7% | No |
| **Whisper finetune v2 (`checkpoint-150`)** | **14.0%** | **8.4%** | **Yes (current best)** |

**Conclusion:** For Bridge2AI read-speech STT, **Whisper large-v3 finetuned on `clean_short_en` (`checkpoint-150`)** is the best result (~**14% dev / 8% test** on `clean_short_en`), far below both the Whisper baseline (~37% / 33%) and the Zipformer path (~57–80%).

Zipformer remains useful as the project’s **early-exit / Libri-trained** acoustic model (16% WER on Libri test-clean) but **does not transfer well** to Bridge2AI read speech without much more data, better references, and stronger decoding (e.g. LM).

---

## 6. Key paths (cluster)

| Artifact | Path |
|----------|------|
| Libri init | `trained_model_zip2layer_100h_mels_only_50ep/mod049-transformer` |
| Zipformer GPU finetune | `bridge2ai_zipformer_full_all_tasks/trained_model_gpu_read/` |
| Zipformer clean-short finetune | `bridge2ai_zipformer_full_all_tasks/trained_model_gpu_read_clean_short/` |
| Whisper baseline (fallback) | `whisper_runs/bridge2ai_read_clean_short_en_baseline_enforced/` |
| Whisper finetune v1 (overfit) | `whisper_runs/bridge2ai_read_clean_short_en_ft/` |
| **Whisper finetune v2 (use this)** | `whisper_runs/bridge2ai_read_clean_short_en_ft_v2/checkpoint-150` |
| Whisper v2 metrics | `whisper_runs/bridge2ai_read_clean_short_en_ft_v2/final_metrics.json` |
| Cluster guide | `bridge2ai_zipformer_cluster_guide.md` |

---

## 7. Recommended next steps

1. **Deploy:** Use `whisper_runs/bridge2ai_read_clean_short_en_ft_v2/checkpoint-150` as the Bridge2AI read STT front-end (`--model_id` pointing at that directory; keep `language=en`, `task=transcribe`).
2. **Archive:** Copy `checkpoint-150`, `final_metrics.json`, and `pred_samples_*.jsonl` off the cluster; delete checkpoints 50–400 except 150 to free disk (~126 GB).
3. **Tighter eval:** filter non-English refs from `clean_short_en` and spot-check Spanish-leak utterances on dev.
4. **Full read set:** eval finetuned Whisper on `read_dev.txt` (with passages) once all passage wavs are on the cluster.
5. **Zipformer (if continued):** LM/beam decoding, dev-WER checkpoint selection, more/better transcripts — not competitive with Whisper v2 for STT today.

---

*Updated Jun 3, 2026 — Irene / early-exit-transformer Bridge2AI read-speech track.*
