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

## Cluster: Whisper CTC fine-tune (SLAM-LLM)

### 1. Pull repo + copy SLAM-LLM

```bash
cd /stek/patsoura/early-exit-transformer
git pull origin irene/ee-ipatsoura-stek-1   # after push

# One-time: place Venkatesh's SLAM-LLM checkout
# (or rsync from Mac Downloads/SLAM-LLM-main)
export SLAM_LLM_ROOT=/stek/patsoura/SLAM-LLM-main
pip install -e "$SLAM_LLM_ROOT"
pip install openai-whisper evaluate
```

Ensure `bridge2ai_adult_wav_v2/` and `bridge2ai_pediatric_wav_v2/` exist on the cluster (same as Whisper CE fine-tune).

If `bridge2ai_norm_ctc/` is missing, run on cluster:

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
