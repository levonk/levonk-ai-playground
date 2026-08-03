#!/usr/bin/env bash
#
# Reasoning Model Runner
# Model: Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled
# Runtime: vLLM — GPTQ-int4 quantization is a vLLM fallback scenario per ADR-202608021744
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./runner-lib.sh
source "$SCRIPT_DIR/runner-lib.sh"

# GPTQ-int4 quant: vLLM has more mature GPTQ support per ADR.
# Override the library default (sglang).
RUNTIME=vllm

# Model-specific configuration
ACCT="${ACCT:-codgician}"
MODEL="${MODEL:-Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-int4}"
HOST_PORT="${HOST_PORT:-8002}"

main() {
  load_env
  check_prerequisites
  setup_huggingface
  clear_caches
  stop_existing_container "$MODEL"

  # Let vLLM auto-detect tokenizer from model config via --trust-remote-code
  # Set GPU memory to 0.9 (large model needs more memory)
  # Enable FlashInfer MoE optimization for better performance
  # Disable CUDA graph memory profiling to avoid memory estimation issues
  GPU_MEMORY_UTILIZATION=0.9
  VLLM_USE_FLASHINFER_MOE_FP16=1
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  #run_container "$ACCT" "$MODEL" "$HOST_PORT" --tokenizer qwen2.Qwen2TokenizerFast
  run_container "$ACCT" "$MODEL" "$HOST_PORT"
}

main "$@"
