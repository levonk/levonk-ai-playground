#!/usr/bin/env bash
#
# Chat Model Runner
# Auto-detects system and selects the appropriate model + runtime.
#
# NVIDIA (DGX Spark GB10):  Qwen3-Next-80B-A3B-Instruct-AWQ-4bit  (SGLang)
# Apple Silicon (Mac Studio): mlx-community Qwen3-Next-80B-A3B 4bit  (MLX)
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
      ACCT="${ACCT:-cyankiwi}"
      MODEL="${MODEL:-Qwen3-Next-80B-A3B-Instruct-AWQ-4bit}"
      HOST_PORT="${HOST_PORT:-8000}"
      RUNTIME="${RUNTIME:-sglang}"
      ;;
    apple-silicon)
      ACCT="${ACCT:-mlx-community}"
      MODEL="${MODEL:-Qwen3-Next-80B-A3B-Instruct-4bit}"
      HOST_PORT="${HOST_PORT:-8000}"
      RUNTIME="${RUNTIME:-mlx}"
      ;;
    vulkan)
      ACCT="${ACCT:-Qwen}"
      MODEL="${MODEL:-Qwen3-Next-80B-A3B-Instruct-GGUF}"
      HOST_PORT="${HOST_PORT:-8000}"
      RUNTIME="${RUNTIME:-vllm}"
      ;;
    *)
      echo "Error: Unsupported system '$system' for chat model." >&2
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

  run_container "$ACCT" "$MODEL" "$HOST_PORT"
}

main "$@"
