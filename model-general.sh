#!/usr/bin/env bash
#
# General/Thinking Model Runner
# Auto-detects system and selects the appropriate model + runtime.
#
# NVIDIA (DGX Spark GB10):  Qwen3-Next-80B-A3B-Thinking-AWQ-4bit  (vLLM, speculative decoding)
# Apple Silicon (Mac Studio): mlx-community Qwen3-Next-80B-A3B-Thinking 4bit  (MLX)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./runner-lib.sh
source "$SCRIPT_DIR/runner-lib.sh"

# Per-system model configuration
configure_model() {
  local system
  system="$(detect_system)"
  echo "Detected system: $system"

  case "$system" in
    nvidia)
      # Uses vLLM-specific --speculative-config flag.
      # SGLang speculative decoding uses different args; migrate when mapped.
      ACCT="${ACCT:-cyankiwi}"
      MODEL="${MODEL:-Qwen3-Next-80B-A3B-Thinking-AWQ-4bit}"
      HOST_PORT="${HOST_PORT:-8003}"
      RUNTIME="${RUNTIME:-vllm}"
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
      ;;
    apple-silicon)
      ACCT="${ACCT:-mlx-community}"
      MODEL="${MODEL:-Qwen3-Next-80B-A3B-Thinking-4bit}"
      HOST_PORT="${HOST_PORT:-8003}"
      RUNTIME="${RUNTIME:-mlx}"
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
      ;;
    vulkan)
      ACCT="${ACCT:-Qwen}"
      MODEL="${MODEL:-Qwen3-Next-80B-A3B-Thinking-GGUF}"
      HOST_PORT="${HOST_PORT:-8003}"
      RUNTIME="${RUNTIME:-vllm}"
      ;;
    *)
      echo "Error: Unsupported system '$system' for general/thinking model." >&2
      echo "Set SYSTEM=nvidia|apple-silicon|vulkan or ACCT/MODEL/RUNTIME manually." >&2
      exit 1
      ;;
  esac
}

main() {
  load_env
  configure_model
  check_prerequisites
  setup_huggingface
  clear_caches
  stop_existing_container "$MODEL"

  local system
  system="$(detect_system)"

  case "$system" in
    nvidia)
      # vLLM-specific arguments with speculative decoding
      run_container "$ACCT" "$MODEL" "$HOST_PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
      ;;
    apple-silicon)
      # MLX: pass max-model-len as extra arg; no speculative-config equivalent
      run_container "$ACCT" "$MODEL" "$HOST_PORT" \
        --max-kv-size "$MAX_MODEL_LEN"
      ;;
    *)
      # Vulkan / CPU fallback: no speculative decoding
      run_container "$ACCT" "$MODEL" "$HOST_PORT"
      ;;
  esac
}

main "$@"
