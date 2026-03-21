#!/usr/bin/env bash
set -euo pipefail

# Run all 9 tasks for a single model.
# Usage:
#   ./scripts/run_all_models_cpu.sh
#
# Notes:
# - CPU-only machine assumed.
# - This writes outputs under ./results/<model_name_with__>/
# - trust_remote_code is hardcoded ON in scripts/run_evaluation.py (no flags needed here).

PY=".venv/bin/python"
TASKS="A1,A2,A3,B1,B2,B3,C1,C2,C3"

# Batch size can be overridden either by env var or first positional arg.
# Examples:
#   ./scripts/run_all_models_cpu.sh           # default
#   ./scripts/run_all_models_cpu.sh 8         # override via arg
#   BATCH_SIZE=8 ./scripts/run_all_models_cpu.sh  # override via env
BATCH_SIZE_DEFAULT=4
BATCH_SIZE="${BATCH_SIZE:-${1:-$BATCH_SIZE_DEFAULT}}"

run_model () {
  local model="$1"

  echo "=== Running model: ${model} (batch_size=${BATCH_SIZE}) ==="
  ${PY} scripts/run_evaluation.py \
    --model "${model}" \
    --tasks "${TASKS}" \
    --batch-size "${BATCH_SIZE}"
}

# 1) Nomic
run_model "nomic-ai/nomic-embed-text-v1"

# 2-5) ChEmbed variants
run_model "BASF-AI/ChEmbed-vanilla"
run_model "BASF-AI/ChEmbed-full"
run_model "BASF-AI/ChEmbed-plug"
run_model "BASF-AI/ChEmbed-prog"
