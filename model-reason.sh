#!/usr/bin/env bash
#
# Reasoning Model Runner
# Auto-detects system and selects the appropriate model + runtime.
#
# NVIDIA (DGX Spark GB10):  Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-int4  (vLLM)
#   NOTE: GPTQ-int4 is a vLLM fallback scenario per ADR-202608021744.
#         FreeToken does not support GPTQ; vLLM is the correct runtime here.
#         If an NVFP4 re-quant becomes available, switch to RUNTIME=freetoken.
# Apple Silicon (Mac Studio): mlx-community Qwen3.5-35B-A3B 4bit  (MLX)
#   NOTE: The Claude-4.6-Opus reasoning distillation may not have an MLX
#         equivalent. The base Qwen3.5-35B-A3B model is used as fallback.
#         Replace MODEL with the MLX-converted distillation if available.
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
      # GPTQ-int4 quant: vLLM has more mature GPTQ support per ADR.
      ACCT="${ACCT:-codgician}"
      MODEL="${MODEL:-Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-int4}"
      HOST_PORT="${HOST_PORT:-8002}"
      RUNTIME="${RUNTIME:-vllm}"
      # Let vLLM auto-detect tokenizer from model config via --trust-remote-code
      # Set GPU memory to 0.9 (large model needs more memory)
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
      # Enable FlashInfer MoE optimization for better performance
      VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-1}"
      # Disable CUDA graph memory profiling to avoid memory estimation issues
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"
      ;;
    apple-silicon)
      # No MLX-converted Claude-4.6-Opus distillation known; use base Qwen3.5-35B-A3B.
      # Replace MODEL with the MLX distillation if/when one is published.
      ACCT="${ACCT:-mlx-community}"
      MODEL="${MODEL:-Qwen3.5-35B-A3B-Instruct-4bit}"
      HOST_PORT="${HOST_PORT:-8002}"
      RUNTIME="${RUNTIME:-mlx}"
      ;;
    vulkan)
      ACCT="${ACCT:-Qwen}"
      MODEL="${MODEL:-Qwen3.5-35B-A3B-Instruct-GGUF}"
      HOST_PORT="${HOST_PORT:-8002}"
      RUNTIME="${RUNTIME:-vllm}"
      ;;
    *)
      echo "Error: Unsupported system '$system' for reasoning model." >&2
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
