#!/bin/bash

set -euo pipefail

AGENT_MODEL="${AGENT_MODEL:-openai/local-qwen25-7b}"
AGENT_API_BASE="${AGENT_API_BASE:-http://127.0.0.1:9000/v1}"
AGENT_API_KEY="${AGENT_API_KEY:-EMPTY}"
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.0}"

USER_MODEL="${USER_MODEL:-gpt-4.1}"
USER_TEMPERATURE="${USER_TEMPERATURE:-0.0}"

NUM_TRIALS="${NUM_TRIALS:-4}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
SEED="${SEED:-300}"

RUN_PREFIX="${RUN_PREFIX:-qwen25_7b_gpt41}"

DOMAINS=(retail airline telecom)

echo "Running tau2 text benchmark for core domains"
echo "  agent model: ${AGENT_MODEL}"
echo "  agent api base: ${AGENT_API_BASE}"
echo "  user model: ${USER_MODEL}"
echo "  num trials: ${NUM_TRIALS}"
echo "  max concurrency: ${MAX_CONCURRENCY}"
echo "  run prefix: ${RUN_PREFIX}"

for domain in "${DOMAINS[@]}"; do
  save_to="${RUN_PREFIX}_${domain}"
  echo
  echo "=== Running domain: ${domain} ==="
  uv run tau2 run \
    --domain "${domain}" \
    --agent-llm "${AGENT_MODEL}" \
    --agent-llm-args "{\"api_base\":\"${AGENT_API_BASE}\",\"api_key\":\"${AGENT_API_KEY}\",\"temperature\":${AGENT_TEMPERATURE}}" \
    --user-llm "${USER_MODEL}" \
    --user-llm-args "{\"temperature\":${USER_TEMPERATURE}}" \
    --task-split-name base \
    --num-trials "${NUM_TRIALS}" \
    --seed "${SEED}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --auto-resume \
    --save-to "${save_to}"
done

echo
echo "Finished all domains."
echo "Summarize with:"
echo "  uv run python scripts/summarize_text_core_benchmark.py --run-prefix ${RUN_PREFIX}"
