#!/usr/bin/env bash
# Matched CE vs CTC Whisper eval on clean_short_en test (239 utts).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export SLAM_LLM_ROOT="${SLAM_LLM_ROOT:-/stek/patsoura/SLAM-LLM-main}"
export PYTHONPATH="${SLAM_LLM_ROOT}/src:${SLAM_LLM_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CE_CKPT="${CE_CKPT:-whisper_runs/bridge2ai_read_clean_short_en_ft_v2/checkpoint-150}"
CTC_CKPT="${CTC_CKPT:-whisper_ctc_runs/bridge2ai_norm_partial_unfreeze/whisper_ctc_partial_unfreeze.pth}"

python3 scripts/run_matched_whisper_asr_eval.py \
  --ce-checkpoint "$CE_CKPT" \
  --ctc-checkpoint "$CTC_CKPT" \
  --slam-llm-root "$SLAM_LLM_ROOT"

echo ""
echo "Results:"
echo "  classification_results/whisper_en_asr_test/ce_matched_test_ref_hyp.tsv"
echo "  classification_results/whisper_en_asr_test/ctc_matched_test_ref_hyp.tsv"
echo "  classification_results/whisper_en_asr_test/matched_asr_comparison.json"
