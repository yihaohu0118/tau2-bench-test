#!/bin/bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/yihao_hyh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-local-qwen25-7b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"
API_KEY="${API_KEY:-EMPTY}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

echo "Starting vLLM OpenAI-compatible server"
echo "  model path: ${MODEL_PATH}"
echo "  model name: ${MODEL_NAME}"
echo "  host: ${HOST}"
echo "  port: ${PORT}"

vllm serve "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --generation-config vllm
