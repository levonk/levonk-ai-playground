#!/usr/bin/env bash
#
# Chat Model Runner
# Model: Qwen3-Next-80B-A3B-Instruct-AWQ-4bit
# Runtime: SGLang (default) — AWQ-4bit is supported by SGLang on GB10
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./runner-lib.sh
source "$SCRIPT_DIR/runner-lib.sh"

# Model-specific configuration
ACCT="${ACCT:-cyankiwi}"
MODEL="${MODEL:-Qwen3-Next-80B-A3B-Instruct-AWQ-4bit}"
HOST_PORT="${HOST_PORT:-8000}"

main() {
  load_env
  check_prerequisites
  setup_huggingface
  clear_caches
  stop_existing_container "$MODEL"

  run_container "$ACCT" "$MODEL" "$HOST_PORT"
}

main "$@"
	

