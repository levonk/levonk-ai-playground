#!/usr/bin/env bash
#
# vLLM General/Thinking Model Runner
# Model: Qwen3-Next-80B-A3B-Thinking-AWQ-4bit
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./vllm-runner-lib.sh
source "$SCRIPT_DIR/vllm-runner-lib.sh"

# Model-specific configuration
ACCT="${ACCT:-cyankiwi}"
MODEL="${MODEL:-Qwen3-Next-80B-A3B-Thinking-AWQ-4bit}"
HOST_PORT="${HOST_PORT:-8003}"

# Additional vLLM parameters for this model
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

main() {
  load_env
  check_prerequisites
  setup_huggingface
  clear_caches
  stop_existing_container "$MODEL"

  # Model-specific vLLM arguments with speculative decoding
  run_vllm_container "$ACCT" "$MODEL" "$HOST_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
}

main "$@"
	

