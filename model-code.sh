#!/usr/bin/env bash
#
# vLLM Code Model Runner
# Model: Qwen3-Coder-Next-AWQ-4bit
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./vllm-runner-lib.sh
source "$SCRIPT_DIR/vllm-runner-lib.sh"

# Model-specific configuration
ACCT="${ACCT:-Intel}"
#MODEL="${MODEL:-Qwen3-Coder-Next-AWQ-4bit}"
MODEL="${MODEL:-Qwen3-Coder-Next-int4-AutoRound}"
HOST_PORT="${HOST_PORT:-8001}"

main() {
  load_env
  check_prerequisites
  setup_huggingface
  clear_caches
  stop_existing_container "$MODEL"

  # Model-specific vLLM arguments
  # Set GPU memory to 0.8 (GPU has 119 GiB, plenty available)
  # Enable FlashInfer MoE optimization for better performance
  GPU_MEMORY_UTILIZATION=0.8
  VLLM_USE_FLASHINFER_MOE_FP16=1
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  run_vllm_container "$ACCT" "$MODEL" "$HOST_PORT"
}

main "$@"
