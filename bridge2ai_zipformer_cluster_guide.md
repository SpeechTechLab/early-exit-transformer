# Bridge2AI read-speech ASR on the cluster

Use this after `git pull` on `/stek/patsoura/early-exit-transformer` (branch `irene/ee-ipatsoura-stek-1`).

## What git contains (small)

- Cleaned manifests: `bridge2ai_zipformer_full_all_tasks/manifests/read_{train,dev,test}.txt`
- Mel path list: `bridge2ai_zipformer_full_all_tasks/manifests/read_mel_paths.txt` (one CSV path per line)
- `subset_meta.tsv` (wav paths + metadata)
- `clean_bridge2ai_manifest.py` (optional re-filter)
- Training code: `train.py`, `inference.py`, `early_zipformer_2layer_exits`, etc.

## What is **not** in git (large — must already exist on cluster)

| Asset | Typical cluster path | Size |
|--------|----------------------|------|
| Libri init checkpoint | `trained_model_zip2layer_100h_mels_only_50ep/mod049-transformer` | ~122 MB |
| Read-speech mel CSVs | `bridge2ai_zipformer_full_all_tasks/mels/*.csv` | ~2.8 GB for current manifests |
| SentencePiece BPE | `sentencepiece/build/libri.bpe-256.model` (+ lex/tok) | few MB |
| Bridge2AI wav (optional) | `bridge2ai_adult_wav_v2/`, `bridge2ai_pediatric_wav_v2/` | only if rebuilding mels |

If mels are missing, either copy from your Mac (see below) or run `build_bridge2ai_zipformer_subset.py` on the cluster when wav trees are available.

## 1. Update code on the cluster

```bash
cd /stek/patsoura/early-exit-transformer
git fetch origin
git checkout irene/ee-ipatsoura-stek-1
git pull origin irene/ee-ipatsoura-stek-1
```

Verify manifests:

```bash
wc -l bridge2ai_zipformer_full_all_tasks/manifests/read_*.txt
# Expect roughly: train ~1370, dev ~254, test ~371
```

Verify Libri checkpoint:

```bash
ls -lh trained_model_zip2layer_100h_mels_only_50ep/mod049-transformer
```

Verify a sample mel exists:

```bash
head -1 bridge2ai_zipformer_full_all_tasks/manifests/read_mel_paths.txt | xargs ls -lh
```

## 2. If mels are missing on the cluster

From your **Mac** (adjust host if needed):

```bash
cd /Users/ipatsoura/early-exit-transformer
rsync -avz --files-from=bridge2ai_zipformer_full_all_tasks/manifests/read_mel_paths.txt \
  ./ root@CLUSTER:/stek/patsoura/early-exit-transformer/
```

Or sync the whole read subset folder if you keep it elsewhere.

## 3. Python env (GPU node)

```bash
cd /stek/patsoura/early-exit-transformer
source .venv/bin/activate   # or create: python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cuda.txt   # or requirements.txt matching your CUDA
python3 -c "import torch; print('cuda', torch.cuda.is_available())"
```

## 4. Train (Libri init → Bridge2AI read-speech)

```bash
cd /stek/patsoura/early-exit-transformer
export CUDA_VISIBLE_DEVICES=0

python3 -u train.py \
  --decoder_mode ctc \
  --model_type early_zipformer_2layer_exits \
  --use_precomputed_features \
  --manifest bridge2ai_zipformer_full_all_tasks/manifests/read_train.txt \
  --n_glottal_features 80 \
  --n_enc_layers_per_exit 2 \
  --n_enc_exits 6 \
  --load_model_path trained_model_zip2layer_100h_mels_only_50ep/mod049-transformer \
  --init_lr 3e-6 \
  --n_epochs 80 \
  --batch_size 16 \
  --n_workers 4 \
  --save_model_dir bridge2ai_zipformer_full_all_tasks/trained_model_gpu_read \
  2>&1 | tee bridge2ai_zipformer_full_all_tasks/logs/train_gpu_read.log
```

Pick the checkpoint with best **dev WER** (not necessarily the last epoch).

## 5. Evaluate on dev and test

```bash
CKPT=bridge2ai_zipformer_full_all_tasks/trained_model_gpu_read/mod016-transformer

python3 inference.py \
  --use_precomputed_features \
  --manifest bridge2ai_zipformer_full_all_tasks/manifests/read_dev.txt \
  --n_glottal_features 80 \
  --decoder_mode ctc \
  --model_type early_zipformer_2layer_exits \
  --n_enc_layers_per_exit 2 \
  --n_enc_exits 6 \
  --load_model_path "$CKPT" \
  --batch_size 16 \
  --n_workers 4 \
  --results_file bridge2ai_zipformer_full_all_tasks/wer_dev_gpu_read.json

python3 inference.py \
  --use_precomputed_features \
  --manifest bridge2ai_zipformer_full_all_tasks/manifests/read_test.txt \
  --n_glottal_features 80 \
  --decoder_mode ctc \
  --model_type early_zipformer_2layer_exits \
  --n_enc_layers_per_exit 2 \
  --n_enc_exits 6 \
  --load_model_path "$CKPT" \
  --batch_size 16 \
  --n_workers 4 \
  --results_file bridge2ai_zipformer_full_all_tasks/wer_test_gpu_read.json
```

Compare against Libri-only baseline on the same dev manifest:

```bash
python3 inference.py \
  --use_precomputed_features \
  --manifest bridge2ai_zipformer_full_all_tasks/manifests/read_dev.txt \
  --n_glottal_features 80 \
  --decoder_mode ctc \
  --model_type early_zipformer_2layer_exits \
  --n_enc_layers_per_exit 2 \
  --n_enc_exits 6 \
  --load_model_path trained_model_zip2layer_100h_mels_only_50ep/mod049-transformer \
  --batch_size 16 \
  --n_workers 4 \
  --results_file bridge2ai_zipformer_full_all_tasks/wer_dev_libri_only.json
```

Report **exit 6** WER from the JSON files.

## 6. Use as speech-to-text front-end

For inference on new utterances, add lines to a manifest:

```text
path/to/mel.csv,reference text optional for eval
```

Extract 80-dim mel CSVs with the same settings as `build_bridge2ai_zipformer_subset.py` (`n_fft=512`, `hop=160`, `win=320`, `sr=16000`, `n_mels=80`).

Then run `inference.py` with `--use_precomputed_features` and your finetuned `--load_model_path`.

## Manifest cleaning (optional)

```bash
python3 clean_bridge2ai_manifest.py \
  --manifest bridge2ai_zipformer_full_all_tasks/manifests/read_train.txt \
  --manifest bridge2ai_zipformer_full_all_tasks/manifests/read_dev.txt \
  --manifest bridge2ai_zipformer_full_all_tasks/manifests/read_test.txt \
  --dedupe-only --in-place
```

## Notes

- Model type must be **`early_zipformer_2layer_exits`** (same as Libri 100h run that reached ~16% WER on test-clean).
- Do **not** use `--append_glottal_features` for this mel-only Bridge2AI path.
- `logs/` and `trained_model_gpu_read/` are gitignored; they stay on the cluster.
