#!/usr/bin/env bash
#
# Code Model Runner
# Auto-detects system and selects the appropriate model + runtime.
#
# NVIDIA (DGX Spark GB10):  Qwen3-Coder-Next-AWQ-4bit  (SGLang)
# Apple Silicon (Mac Studio): mlx-community Qwen3-Coder-Next 4bit  (MLX)
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
      MODEL="${MODEL:-Qwen3-Coder-Next-AWQ-4bit}"
      HOST_PORT="${HOST_PORT:-8001}"
      RUNTIME="${RUNTIME:-sglang}"
      # Set GPU memory to 0.8 (GPU has 119 GiB, plenty available)
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
      # vLLM-specific optimizations (only used when RUNTIME=vllm)
      VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-1}"
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"
      ;;
    apple-silicon)
      ACCT="${ACCT:-mlx-community}"
      MODEL="${MODEL:-Qwen3-Coder-Next-4bit}"
      HOST_PORT="${HOST_PORT:-8001}"
      RUNTIME="${RUNTIME:-mlx}"
      ;;
    vulkan)
      ACCT="${ACCT:-Qwen}"
      MODEL="${MODEL:-Qwen3-Coder-Next-GGUF}"
      HOST_PORT="${HOST_PORT:-8001}"
      RUNTIME="${RUNTIME:-vllm}"
      ;;
    *)
      echo "Error: Unsupported system '$system' for code model." >&2
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
