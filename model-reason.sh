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
  run_vllm_container "$ACCT" "$MODEL" "$HOST_PORT"
}

main "$@"

