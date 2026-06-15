#!/usr/bin/env bash
# Fine-tune Whisper with CTC loss on Bridge2AI NORM data (SLAM-LLM scripts).
#
# Prerequisites on the cluster:
#   1. git pull early-exit-transformer (bridge2ai_norm_ctc/ jsonl + wav symlinks)
#   2. SLAM-LLM checkout at SLAM_LLM_ROOT (default below)
#   3. pip install -e "$SLAM_LLM_ROOT" plus requirements (whisper, transformers, evaluate)
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   bash scripts/run_whisper_ctc_bridge2ai.sh partial_unfreeze
#   bash scripts/run_whisper_ctc_bridge2ai.sh frozen_backbone

set -euo pipefail

MODE="${1:-partial_unfreeze}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLAM_LLM_ROOT="${SLAM_LLM_ROOT:-/stek/patsoura/SLAM-LLM-main}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/bridge2ai_norm_ctc}"
TRAIN_JSONL="${TRAIN_JSONL:-$DATA_ROOT/jsonl/train.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-$DATA_ROOT/jsonl/dev.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/whisper_ctc_runs/bridge2ai_norm_${MODE}}"

export PYTHONPATH="${SLAM_LLM_ROOT}/src:${SLAM_LLM_ROOT}:${PYTHONPATH:-}"
export WANDB_DISABLED=true

case "$MODE" in
  frozen_backbone)
    SCRIPT="${SLAM_LLM_ROOT}/src/whisper_ctc/train_frozen_backbone.py"
    ;;
  partial_unfreeze)
    SCRIPT="${SLAM_LLM_ROOT}/src/whisper_ctc/train_partial_unfreeze.py"
    ;;
  *)
    echo "Unknown mode: $MODE (use frozen_backbone or partial_unfreeze)" >&2
    exit 1
    ;;
esac

if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing SLAM-LLM script: $SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Missing train jsonl: $TRAIN_JSONL" >&2
  echo "Run: python3 prepare_bridge2ai_norm_ctc_data.py --train-manifest .../train.norm.txt" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
python3 - "$SCRIPT" "$TRAIN_JSONL" "$EVAL_JSONL" "$OUTPUT_DIR" <<'PY'
import runpy
import sys

script, train_jsonl, eval_jsonl, output_dir = sys.argv[1:5]

# Patch dataset paths and output_dir in the SLAM-LLM training scripts.
import pathlib

text = pathlib.Path(script).read_text(encoding="utf-8")
text = text.replace(
    '"/stek/vskadandale/librispeech_jsonl/train-libri.jsonl"',
    repr(train_jsonl),
)
text = text.replace(
    '"/stek/vskadandale/librispeech_jsonl/librispeech_dev_other.jsonl"',
    repr(eval_jsonl),
)
text = text.replace(
    'output_dir="./exp/whisper_ctc_frozen_backbone"',
    f"output_dir={repr(output_dir)}",
)
text = text.replace(
    'output_dir="./exp/whisper_ctc_partial_unfreeze"',
    f"output_dir={repr(output_dir)}",
)
text = text.replace(
    '"./exp/whisper_ctc_frozen_backbone/whisper_ctc_frozen_backbone.pth"',
    f'"{output_dir}/whisper_ctc_frozen_backbone.pth"',
)
text = text.replace(
    '"./exp/whisper_ctc_partial_unfreeze/whisper_ctc_partial_unfreeze.pth"',
    f'"{output_dir}/whisper_ctc_partial_unfreeze.pth"',
)

ns = {"__name__": "__main__"}
exec(compile(text, script, "exec"), ns)
PY

echo "Done. Checkpoints under: $OUTPUT_DIR"
