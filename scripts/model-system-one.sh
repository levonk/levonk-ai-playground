#!/usr/bin/env bash
#
# System One Scorer — calibrated single-pass decision scorer
# (pngwn/system-one-qwen3.5-4b-scorer-v2b)
#
# A "System One"-style scorer: a LoRA adapter on Qwen/Qwen3.5-4B-Base with a
# scalar sequence-classification head. Each option of a typed question
# (yes/no, choice, score) gets a scalar score in a single forward pass, then
# the scores are softmaxed per question with a fitted temperature (2.35).
# There is no autoregressive generation — the output space is exactly the
# option set the caller supplies. Good for fast routing/triage decisions at
# single-pass latency (~128 ms per 5-option query on the training box).
#
# Why a native Python process instead of SGLang/vLLM/MLX:
#   - The scoring head lives in the adapter's modules_to_save (PEFT SEQ_CLS),
#     which the generative runtimes do not load — vLLM/SGLang would serve the
#     base model as a text generator, not a scorer
#   - The scorer needs only torch + transformers + peft, so a small stdlib
#     HTTP server (system-one-server.py, in this repo) exposes POST /score
#   - Precedent: FreeToken and MLX runtimes also run as native Python
#
# Requirements (host, all platforms):
#   pip install torch transformers peft
#
# License note: the adapter is CC-BY-NC-4.0 — non-commercial use only.
#
# Port: 8007 (unique, allows co-location with the LLM/TTS/STT services)
# API:  GET  /health     — liveness + model id
#       GET  /v1/models  — OpenAI-style model list (for test-query probes)
#       POST /score      — {"state": "...", "question": "...", "options": [...]}
#
# Usage:
#   ./scripts/model-system-one.sh                     # start scorer on port 8007
#   HOST_PORT=9007 ./scripts/model-system-one.sh      # custom port
#   S1_TEMPERATURE=1.0 ./scripts/model-system-one.sh  # disable calibration
#
# Test:
#   curl -X POST http://localhost:8007/score \
#     -H "Content-Type: application/json" \
#     -d '{"state":"Customer: my shoes arrived in the wrong size",
#          "question":"Which team should handle this?",
#          "options":["returns","shipping","billing"]}'
#
# Environment variables (see system-one-server.py docstring for full list):
#   S1_BASE_MODEL    — base checkpoint (default: Qwen/Qwen3.5-4B-Base)
#   S1_ADAPTER       — adapter repo (default: pngwn/...-scorer-v2b)
#   S1_MAX_LEN       — max sequence length (default: 384, matches training)
#   S1_TEMPERATURE   — calibration temperature (default: 2.35, fitted on val)
#   S1_BATCH_SIZE    — options per forward pass (default: 8)
#   S1_DEVICE        — cuda|mps|cpu (default: auto)
#   S1_DTYPE         — bf16|fp16|fp32 (default: bf16 on GPU/MPS, fp32 on CPU)
#
# See:
#   - https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer-v2b (model card)
#   - https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer (training code:
#     system_one.py — the encode/score logic in the server mirrors it)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The scorer runs as a native Python process on every platform (same
# precedent as the FreeToken and MLX runtimes). RUNTIME=native keeps the
# library's Docker dispatch/prereq checks out of the way and makes
# stop_existing_container() use the pgrep path instead of `docker ps`.
RUNTIME="${RUNTIME:-native}"
# shellcheck source=../libexec/runner-lib.sh
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/libexec/runner-lib.sh"

# Per-system configuration
configure_model() {
  local system
  system="$(detect_system)"
  echo "Detected system: $system"

  case "$system" in
    nvidia|apple-silicon|vulkan|cpu)
      # Same native deployment on every system — the server auto-selects
      # cuda/mps/cpu. ~8.5 GB in bf16 leaves plenty of headroom for
      # co-located models on GB10.
      HOST_PORT="${HOST_PORT:-8007}"
      # Name used by stop_existing_container()'s pgrep match; must appear in
      # the process command line (the server script's filename does).
      S1_CONTAINER_NAME="${S1_CONTAINER_NAME:-system-one-server.py}"
      ;;
    *)
      echo "Error: Unsupported system '$system' for the System One scorer." >&2
      exit 1
      ;;
  esac
}

main() {
  load_env
  configure_model

  # Dep check — mirrors the CLI checks in check_prerequisites() for the
  # other runtimes (which don't apply to RUNTIME=native).
  command -v python3 >/dev/null 2>&1 || {
    echo "Error: python3 not found." >&2
    exit 1
  }
  if ! python3 -c "import torch, transformers, peft" 2>/dev/null; then
    echo "Error: missing Python dependencies. Install with:" >&2
    echo "  pip install torch transformers peft" >&2
    exit 1
  fi

  setup_huggingface
  clear_caches
  stop_existing_container "$S1_CONTAINER_NAME"

  run_native "$S1_CONTAINER_NAME" "$HOST_PORT" \
    python3 "$REPO_ROOT/libexec/system-one-server.py" --port "$HOST_PORT"
}

main "$@"
