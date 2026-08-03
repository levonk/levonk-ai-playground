#!/usr/bin/env bash
#
# General/Thinking Model Runner
# Model: Qwen3-Next-80B-A3B-Thinking-AWQ-4bit
# Runtime: vLLM — uses vLLM-specific --speculative-config flag
#          (SGLang speculative decoding uses different args; migrate when mapped)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./runner-lib.sh
source "$SCRIPT_DIR/runner-lib.sh"

# This script uses vLLM-specific speculative decoding args.
# Override the library default (sglang) until SGLang equivs are mapped.
RUNTIME=vllm

# Model-specific configuration
ACCT="${ACCT:-cyankiwi}"
MODEL="${MODEL:-Qwen3-Next-80B-A3B-Thinking-AWQ-4bit}"
HOST_PORT="${HOST_PORT:-8003}"

# Additional parameters for this model
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

main() {
  load_env
  check_prerequisites
  setup_huggingface
  clear_caches
  stop_existing_container "$MODEL"

  # vLLM-specific arguments with speculative decoding
  run_container "$ACCT" "$MODEL" "$HOST_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
}

main "$@"
	

