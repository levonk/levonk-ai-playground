#!/usr/bin/env bash
#
# Code Model Runner
# Model: Qwen3-Coder-Next-AWQ-4bit
# Runtime: SGLang (default) — AWQ-4bit is supported by SGLang on GB10
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./runner-lib.sh
source "$SCRIPT_DIR/runner-lib.sh"

# Model-specific configuration
ACCT="${ACCT:-cyankiwi}"
MODEL="${MODEL:-Qwen3-Coder-Next-AWQ-4bit}"
#ACCT="${ACCT:-Intel}"
#MODEL="${MODEL:-Qwen3-Coder-Next-int4-AutoRound}"
HOST_PORT="${HOST_PORT:-8001}"

main() {
  load_env
  check_prerequisites
  setup_huggingface
  clear_caches
  stop_existing_container "$MODEL"

  # Set GPU memory to 0.8 (GPU has 119 GiB, plenty available)
  GPU_MEMORY_UTILIZATION=0.8
  # vLLM-specific optimizations (only used when RUNTIME=vllm)
  VLLM_USE_FLASHINFER_MOE_FP16=1
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  run_container "$ACCT" "$MODEL" "$HOST_PORT"
}

main "$@"
