#!/usr/bin/env bash
# Transfer Bridge2AI wavs needed for NORM Whisper CTC from Mac → FBK container.
#
# Run from anywhere; script cds to repo root automatically.
#
# Prerequisites:
#   1. cd not required, but wav trees must exist under repo:
#        bridge2ai_adult_wav_v2/
#        bridge2ai_pediatric_wav_v2/
#   2. SSH works:
#        ssh -J ipatsoura@jumpsso.fbk.eu stek@digis-rf4421.fbk.eu echo ok
#      If "Permission denied (publickey)", run: ssh-add ~/.ssh/id_rsa
#
# Usage:
#   bash scripts/transfer_norm_ctc_wavs_to_cluster.sh
#   bash scripts/transfer_norm_ctc_wavs_to_cluster.sh --full-dirs   # ~18 GB, simpler

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

JUMP="${JUMP:-ipatsoura@jumpsso.fbk.eu}"
HOST="${HOST:-stek@digis-rf4421.fbk.eu}"
CONTAINER="${CONTAINER:-ee-ipatsoura-stek-4}"
REMOTE_ROOT="${REMOTE_ROOT:-/stek/patsoura/early-exit-transformer}"
PATH_LIST="${PATH_LIST:-bridge2ai_norm_ctc/manifests/norm_ctc_wav_paths.txt}"

if [[ "${1:-}" == "--full-dirs" ]]; then
  echo "Transferring full wav directories (~18 GB). This may take a while..."
  COPYFILE_DISABLE=1 tar czf - bridge2ai_adult_wav_v2 bridge2ai_pediatric_wav_v2 \
    | ssh -J "$JUMP" "$HOST" \
      "docker exec -i $CONTAINER tar -xzf - -C $REMOTE_ROOT"
  echo "Done. On container run:"
  echo "  ls $REMOTE_ROOT/bridge2ai_adult_wav_v2 | wc -l"
  echo "  ls $REMOTE_ROOT/bridge2ai_pediatric_wav_v2 | wc -l"
  exit 0
fi

if [[ ! -f "$PATH_LIST" ]]; then
  echo "Missing $PATH_LIST — run: python3 export_norm_ctc_wav_paths.py" >&2
  exit 1
fi

MISSING_LIST="$(mktemp)"
EXISTING_LIST="$(mktemp)"
trap 'rm -f "$MISSING_LIST" "$EXISTING_LIST"' EXIT

while IFS= read -r p; do
  [[ -z "$p" ]] && continue
  if [[ -f "$p" ]]; then
    echo "$p" >> "$EXISTING_LIST"
  else
    echo "$p" >> "$MISSING_LIST"
  fi
done < "$PATH_LIST"

N_EXIST=$(wc -l < "$EXISTING_LIST" | tr -d ' ')
N_MISS=$(wc -l < "$MISSING_LIST" | tr -d ' ')
echo "Repo root: $REPO_ROOT"
echo "Wav paths in list: $((N_EXIST + N_MISS))"
echo "Existing locally:  $N_EXIST"
echo "Missing locally:   $N_MISS"

if [[ "$N_MISS" -gt 0 ]]; then
  echo "[error] $N_MISS wav files missing on Mac. First few:" >&2
  head -5 "$MISSING_LIST" >&2
  echo "Regenerate wavs with bridge2ai_spectrogram_to_waveform.py if needed." >&2
  exit 1
fi

echo "Testing SSH..."
ssh -J "$JUMP" "$HOST" "docker exec $CONTAINER echo container_ok"

echo "Streaming ~8.8 GB to $CONTAINER:$REMOTE_ROOT ..."
COPYFILE_DISABLE=1 tar czf - -T "$EXISTING_LIST" \
  | ssh -J "$JUMP" "$HOST" \
    "docker exec -i $CONTAINER tar -xzf - -C $REMOTE_ROOT"

echo "Done. Verify in container:"
echo "  docker exec -it $CONTAINER bash -lc 'ls $REMOTE_ROOT/bridge2ai_adult_wav_v2 | wc -l; ls $REMOTE_ROOT/bridge2ai_pediatric_wav_v2 | wc -l'"
