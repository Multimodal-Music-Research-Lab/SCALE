#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
export PYTHONPATH="$SCALE_REPO/third_party/soulx_singer${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCALE_REPO/third_party/soulx_singer"
"$SOULX_PY" "$SCALE_REPO/tools/extract_lyrics.py" --sat_model "$SAT_MODEL" --sat_tokenizer "$SAT_TOKENIZER" "$@"
