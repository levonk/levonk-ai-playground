#!/usr/bin/env bash
#
# vLLM Reasoning Model Runner
# Model: Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./vllm-runner-lib.sh
source "$SCRIPT_DIR/vllm-runner-lib.sh"

# Model-specific configuration
ACCT="${ACCT:-hesamation}"
MODEL="${MODEL:-Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled}"
HOST_PORT="${HOST_PORT:-8002}"

main() {
  load_env
  check_prerequisites
  setup_huggingface
  clear_caches
  stop_existing_container "$MODEL"

  # Model-specific vLLM arguments
  # Let vLLM auto-detect tokenizer from model config via --trust-remote-code
  # Set GPU memory to 0.9 (large model needs more memory)
  # Enable FlashInfer MoE optimization for better performance
  # Disable CUDA graph memory profiling to avoid memory estimation issues
  GPU_MEMORY_UTILIZATION=0.9
  VLLM_USE_FLASHINFER_MOE_FP16=1
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  run_vllm_container "$ACCT" "$MODEL" "$HOST_PORT"
}

main "$@"
