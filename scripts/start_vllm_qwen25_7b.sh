#!/bin/bash

set -euo pipefail

MODEL_PATH="${VLLM_MODEL_PATH:-${MODEL_PATH:-/home/yihao_hyh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct}}"
MODEL_NAME="${VLLM_MODEL_NAME:-${MODEL_NAME:-local-qwen25-7b}}"
VLLM_BIND_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_BIND_PORT="${VLLM_PORT:-9000}"
API_KEY="${VLLM_API_KEY:-${API_KEY:-EMPTY}}"
DTYPE="${VLLM_DTYPE:-${DTYPE:-auto}}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-${GPU_MEMORY_UTILIZATION:-0.9}}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-32768}}"

# Hugging Face cache layout stores the real model under snapshots/<hash>.
# If the provided path is the cache repo root, resolve it automatically.
if [[ -d "${MODEL_PATH}/snapshots" && ! -f "${MODEL_PATH}/config.json" ]]; then
  RESOLVED_SNAPSHOT="$(find "${MODEL_PATH}/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [[ -n "${RESOLVED_SNAPSHOT}" ]]; then
    MODEL_PATH="${RESOLVED_SNAPSHOT}"
  fi
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "config.json not found under model path: ${MODEL_PATH}" >&2
  exit 1
fi

echo "Starting vLLM OpenAI-compatible server"
echo "  model path: ${MODEL_PATH}"
echo "  model name: ${MODEL_NAME}"
echo "  host: ${VLLM_BIND_HOST}"
echo "  port: ${VLLM_BIND_PORT}"

vllm serve "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --host "${VLLM_BIND_HOST}" \
  --port "${VLLM_BIND_PORT}" \
  --api-key "${API_KEY}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --generation-config vllm
