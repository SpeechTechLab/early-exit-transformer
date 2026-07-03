# Bridge2AI NORM → Whisper CTC (SLAM-LLM)

## Data layout (after `prepare_bridge2ai_norm_ctc_data.py`)

| Path | Contents |
|------|----------|
| `bridge2ai_norm_ctc/wav/` | **32,827** symlinks to reconstructed 16 kHz wavs (`22,881` train + dev/test) |
| `bridge2ai_norm_ctc/jsonl/train.jsonl` | 22,881 utts from `train.norm.txt` |
| `bridge2ai_norm_ctc/jsonl/dev.jsonl` | 3,846 utts (subset_meta dev, normalized text) |
| `bridge2ai_norm_ctc/jsonl/test.jsonl` | 6,100 utts (subset_meta test, normalized text) |

Wavs are **not** re-synthesized here; they point to existing Griffin–Lim reconstructions under `bridge2ai_adult_wav_v2/` and `bridge2ai_pediatric_wav_v2/`.

Regenerate locally:

```bash
python3 prepare_bridge2ai_norm_ctc_data.py \
  --train-manifest bridge2ai_norm_ctc/manifests/train.norm.txt \
  --output-dir bridge2ai_norm_ctc
```

Use `--copy` if you need real files for `tar`/`rsync` to the cluster (symlinks break when copied naively).

## Cluster: copy wavs first (required)

The cluster usually has **mels** but not the full **wav** trees. If `prepare_bridge2ai_norm_ctc_data.py` reports thousands of missing wavs and only ~900 train utts, copy wavs from your Mac first.

**On Mac** (must run from repo root — paths in the list are relative):

```bash
cd /Users/ipatsoura/early-exit-transformer

# Test SSH first (fix ssh-add if this fails)
ssh -J ipatsoura@jumpsso.fbk.eu stek@digis-rf4421.fbk.eu echo ok

# Recommended helper (~8.8 GB, only NORM-required wavs)
bash scripts/transfer_norm_ctc_wavs_to_cluster.sh

# Or transfer both full wav trees (~18 GB, simpler)
bash scripts/transfer_norm_ctc_wavs_to_cluster.sh --full-dirs
```

Manual equivalent:

```bash
cd /Users/ipatsoura/early-exit-transformer
python3 export_norm_ctc_wav_paths.py

COPYFILE_DISABLE=1 tar czf - -T bridge2ai_norm_ctc/manifests/norm_ctc_wav_paths.txt \
  | ssh -J ipatsoura@jumpsso.fbk.eu stek@digis-rf4421.fbk.eu \
    'docker exec -i ee-ipatsoura-stek-4 tar -xzf - -C /stek/patsoura/early-exit-transformer'
```

If tar prints `Cannot stat` for many files, you are **not** in `early-exit-transformer` (you ran from `~`).
If SSH prints `Permission denied (publickey)`, run `ssh-add ~/.ssh/id_rsa` and retry.

Expect **~32k wav files** (~several GB). After transfer, on the cluster:

```bash
cd /stek/patsoura/early-exit-transformer
python3 prepare_bridge2ai_norm_ctc_data.py \
  --train-manifest bridge2ai_norm_ctc/manifests/train.norm.txt \
  --output-dir bridge2ai_norm_ctc
# expect: train_utts 22881, dev_utts 3846, test_utts 6100
```

## Cluster: Whisper CTC fine-tune (SLAM-LLM)

### 1. Pull repo + copy SLAM-LLM

```bash
cd /stek/patsoura/early-exit-transformer
git pull origin irene/ee-ipatsoura-stek-1
```

**SLAM-LLM is not in this repo.** Copy it once from your Mac (path must exist before `pip install -e`):

```bash
# On Mac:
rsync -avz -e "ssh -J ipatsoura@jumpsso.fbk.eu" \
  /Users/ipatsoura/Downloads/SLAM-LLM-main/ \
  stek@digis-rf4421.fbk.eu:/stek/patsoura/SLAM-LLM-main/
```

**Inside the container:**

```bash
export SLAM_LLM_ROOT=/stek/patsoura/SLAM-LLM-main
ls "$SLAM_LLM_ROOT/pyproject.toml"   # must exist

cd "$SLAM_LLM_ROOT"
pip install -e .
pip install openai-whisper evaluate

# or without editable install:
# pip install -r requirements.txt openai-whisper evaluate
# export PYTHONPATH="$SLAM_LLM_ROOT/src:$SLAM_LLM_ROOT:$PYTHONPATH"
```

Ensure `bridge2ai_adult_wav_v2/` and `bridge2ai_pediatric_wav_v2/` exist on the cluster (see wav copy section above).

If `bridge2ai_norm_ctc/jsonl/` is missing after wav copy, run on cluster:

```bash
python3 prepare_bridge2ai_norm_ctc_data.py \
  --train-manifest bridge2ai_norm_ctc/manifests/train.norm.txt \
  --output-dir bridge2ai_norm_ctc
```

### 2. Train (recommended: partial unfreeze)

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_whisper_ctc_bridge2ai.sh partial_unfreeze
```

Frozen backbone only (CTC head):

```bash
bash scripts/run_whisper_ctc_bridge2ai.sh frozen_backbone
```

Checkpoints: `whisper_ctc_runs/bridge2ai_norm_partial_unfreeze/` (HF Trainer format + final `.pth`).

### 3. Inference / WER on test

```bash
export PYTHONPATH="$SLAM_LLM_ROOT/src:$SLAM_LLM_ROOT:$PYTHONPATH"
cd "$SLAM_LLM_ROOT/src/whisper_ctc"
python3 inference.py   # edit checkpoint_path + test_jsonl to bridge2ai_norm_ctc/jsonl/test.jsonl
```

## JSONL format (SLAM-LLM)

Each line:

```json
{"source": "/abs/path/to.wav", "target": "how hard did he hit him"}
```

Targets are lowercase, punctuation stripped (matches `train.norm.txt`).
