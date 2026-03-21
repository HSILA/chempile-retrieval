#!/usr/bin/env bash
set -euo pipefail

# Run all 9 tasks for a single model.
# Usage:
#   ./scripts/run_all_models_cpu.sh
#
# Notes:
# - CPU-only machine assumed.
# - This writes outputs under ./results/<model_name_with__>/
# - For models that require it, we pass --trust-remote-code.

PY=".venv/bin/python"
TASKS="A1,A2,A3,B1,B2,B3,C1,C2,C3"

run_model () {
  local model="$1"
  local batch="$2"
  local trust_flag="${3:-}"

  echo "=== Running model: ${model} (batch_size=${batch}) ==="
  ${PY} scripts/run_evaluation.py \
    --model "${model}" \
    --tasks "${TASKS}" \
    --batch-size "${batch}" \
    ${trust_flag}
}

# 1) Nomic (requires trust_remote_code)
run_model "nomic-ai/nomic-embed-text-v1" 4

# 2-5) ChEmbed variants
run_model "BASF-AI/ChEmbed-vanilla" 4
run_model "BASF-AI/ChEmbed-full" 4
run_model "BASF-AI/ChEmbed-plug" 4
run_model "BASF-AI/ChEmbed-prog" 4
