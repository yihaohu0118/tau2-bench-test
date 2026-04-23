#!/bin/bash

set -euo pipefail

if [[ -f ".env" ]]; then
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    [[ "${line}" != *=* ]] && continue

    key="${line%%=*}"
    value="${line#*=}"

    # Skip placeholder template values like <your_key_here>.
    if [[ "${value}" =~ ^\<.*\>$ ]]; then
      continue
    fi

    export "${key}=${value}"
  done < ".env"
fi

AGENT_MODEL="${AGENT_MODEL:-openai/local-qwen25-7b}"
AGENT_API_BASE="${AGENT_API_BASE:-http://127.0.0.1:9000/v1}"
AGENT_API_KEY="${AGENT_API_KEY:-EMPTY}"
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.0}"

USER_MODEL="${USER_MODEL:-gpt-4.1}"
USER_TEMPERATURE="${USER_TEMPERATURE:-0.0}"
USER_API_KEY="${USER_API_KEY:-}"
USER_API_BASE="${USER_API_BASE:-}"

NUM_TRIALS="${NUM_TRIALS:-4}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
SEED="${SEED:-300}"

RUN_PREFIX="${RUN_PREFIX:-qwen25_7b_gpt41}"

DOMAINS=(retail airline telecom)

require_env_var() {
  local env_name="$1"
  local reason="$2"
  if [[ -z "${!env_name:-}" ]]; then
    echo "ERROR: ${reason}. Missing environment variable: ${env_name}" >&2
    exit 1
  fi
}

ORIGINAL_USER_MODEL="${USER_MODEL}"

if [[ "${USER_MODEL}" == openrouter/* ]]; then
  require_env_var "OPENROUTER_API_KEY" \
    "USER_MODEL=${USER_MODEL} requires an OpenRouter API key"
  if [[ -z "${USER_API_KEY}" ]]; then
    USER_API_KEY="${OPENROUTER_API_KEY}"
  fi
  if [[ -z "${USER_API_BASE}" ]]; then
    USER_API_BASE="https://openrouter.ai/api/v1"
  fi
  # Route OpenRouter through its OpenAI-compatible endpoint using the raw
  # OpenRouter model id, e.g. openai/gpt-4.1.
  USER_MODEL="${USER_MODEL#openrouter/}"
elif [[ "${USER_MODEL}" == gpt-* || "${USER_MODEL}" == openai/* ]]; then
  if [[ -z "${USER_API_KEY}" ]]; then
    require_env_var "OPENAI_API_KEY" \
      "USER_MODEL=${USER_MODEL} requires an OpenAI API key"
    USER_API_KEY="${OPENAI_API_KEY}"
  fi
fi

USER_LLM_ARGS="{\"api_key\":\"${USER_API_KEY}\",\"temperature\":${USER_TEMPERATURE}"
if [[ -n "${USER_API_BASE}" ]]; then
  USER_LLM_ARGS="${USER_LLM_ARGS},\"api_base\":\"${USER_API_BASE}\""
fi
USER_LLM_ARGS="${USER_LLM_ARGS}}"

if [[ -n "${USER_API_BASE}" && -n "${USER_API_KEY}" ]]; then
  echo "Running user API preflight against ${USER_API_BASE}/models"
  http_code="$(
    curl -sS -o /tmp/tau2_user_models_check.json -w "%{http_code}" \
      -H "Authorization: Bearer ${USER_API_KEY}" \
      "${USER_API_BASE}/models"
  )"
  if [[ "${http_code}" != "200" ]]; then
    echo "ERROR: user API preflight failed with HTTP ${http_code}" >&2
    cat /tmp/tau2_user_models_check.json >&2 || true
    exit 1
  fi
fi

echo "Running tau2 text benchmark for core domains"
echo "  agent model: ${AGENT_MODEL}"
echo "  agent api base: ${AGENT_API_BASE}"
echo "  user model: ${USER_MODEL}"
echo "  user model input: ${ORIGINAL_USER_MODEL}"
echo "  user api base: ${USER_API_BASE:-<default>}"
echo "  openai key loaded: $([[ -n "${OPENAI_API_KEY:-}" ]] && echo yes || echo no)"
echo "  openrouter key loaded: $([[ -n "${OPENROUTER_API_KEY:-}" ]] && echo yes || echo no)"
echo "  user api key wired: $([[ -n "${USER_API_KEY:-}" ]] && echo yes || echo no)"
echo "  num trials: ${NUM_TRIALS}"
echo "  max concurrency: ${MAX_CONCURRENCY}"
echo "  run prefix: ${RUN_PREFIX}"

for domain in "${DOMAINS[@]}"; do
  save_to="${RUN_PREFIX}_${domain}"
  echo
  echo "=== Running domain: ${domain} ==="
  OPENAI_API_KEY="${USER_API_KEY:-${OPENAI_API_KEY:-}}" \
  OPENAI_API_BASE="${USER_API_BASE:-${OPENAI_API_BASE:-}}" \
  OPENROUTER_API_KEY="${USER_API_KEY:-${OPENROUTER_API_KEY:-}}" \
  uv run tau2 run \
    --domain "${domain}" \
    --agent-llm "${AGENT_MODEL}" \
    --agent-llm-args "{\"api_base\":\"${AGENT_API_BASE}\",\"api_key\":\"${AGENT_API_KEY}\",\"temperature\":${AGENT_TEMPERATURE}}" \
    --user-llm "${USER_MODEL}" \
    --user-llm-args "${USER_LLM_ARGS}" \
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
