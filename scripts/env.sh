#!/usr/bin/env bash
set -euo pipefail
export SCALE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$SCALE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export SCALE_PY="${SCALE_PY:-python}"
export SCALE_LONGFORMER_PATH="${SCALE_LONGFORMER_PATH:-$SCALE_REPO/models/longformer-base-4096}"
export SOULX_PY="${SOULX_PY:-python}"
export SOULX_MODEL_ROOT="${SOULX_MODEL_ROOT:-$SCALE_REPO/models/SoulX-Singer-Preprocess}"
export SAT_MODEL="${SAT_MODEL:-$SCALE_REPO/models/sat-3l-sm}"
export SAT_TOKENIZER="${SAT_TOKENIZER:-$SCALE_REPO/models/xlm-roberta-base}"
export MUQ_CHECKPOINT="${MUQ_CHECKPOINT:-OpenMuQ/MuQ-large-msd-iter}"
export TOKENIZERS_PARALLELISM=false
