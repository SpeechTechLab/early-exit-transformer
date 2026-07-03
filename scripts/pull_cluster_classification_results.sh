#!/usr/bin/env bash
# Copy classification summary CSVs from the FBK cluster (small files only).
set -euo pipefail

REPO="${REPO:-/Users/ipatsoura/early-exit-transformer}"
REMOTE="${REMOTE:-stek@digis-rf4421.fbk.eu}"
JUMP="${JUMP:-ipatsoura@jumpsso.fbk.eu}"
CONTAINER="${CONTAINER:-ee-ipatsoura-stek-4}"
REMOTE_REPO="${REMOTE_REPO:-/stek/patsoura/early-exit-transformer}"

RUNS=(
  run_20260605_100410
  run_20260607_130745
)

mkdir -p "${REPO}/classification_results"

for run in "${RUNS[@]}"; do
  dest="${REPO}/classification_results/${run}"
  mkdir -p "${dest}"
  echo "Pulling ${run}/global_summary_metrics.csv ..."
  ssh -J "${JUMP}" "${REMOTE}" \
    "docker exec ${CONTAINER} cat ${REMOTE_REPO}/classification_results/${run}/global_summary_metrics.csv" \
    > "${dest}/global_summary_metrics.csv"
done

echo "Done. Regenerate tables with:"
echo "  cd ${REPO} && python3 build_ddk_f1_tables.py --metric sample --write-md --write-latex"
