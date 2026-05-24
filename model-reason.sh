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
  # Add --tokenizer to fix missing tokenizer issue
  # Set GPU memory to 0.9 (large model needs more memory)
  # Enable FlashInfer MoE optimization for better performance
  # Disable CUDA graph memory profiling to avoid memory estimation issues
<<<<<<< HEAD
  GPU_MEMORY_UTILIZATION=0.9
=======
  GPU_MEMORY_UTILIZATION=0.65
>>>>>>> a26b2c74ce66679c086bb2f06efc122305021552
  VLLM_USE_FLASHINFER_MOE_FP16=1
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  run_vllm_container "$ACCT" "$MODEL" "$HOST_PORT" --tokenizer Qwen/Qwen2.5-32B-Instruct
}

main "$@"
