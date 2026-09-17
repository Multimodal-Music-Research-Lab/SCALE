#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
CONFIG="${SCALE_CONFIG:-$SCALE_REPO/configs/train.yaml}"
"$SCALE_PY" -m scale.train --config "$CONFIG" "$@"
