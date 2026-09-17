#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$SCALE_PY" -m scale.preprocess ssl --provider muq --muq-checkpoint "$MUQ_CHECKPOINT" "$@"
