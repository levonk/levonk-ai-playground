#!/usr/bin/env bash
#
# Lite Model Runner — Lightweight Companion Model
#
# A small, fast model for routine tasks that don't need the full depth of
# the 80B primary models. Use this for:
#   - Formatting / rewriting text
#   - Simple Q&A ("what does this error mean?")
#   - JSON/YAML formatting
#   - Summarization in one or two sentences
#   - Throwaway chats
#
# The lite model uses ~2.5 GB of VRAM (AWQ-4bit), leaving the rest for the
# primary model's KV cache. On GB10 128GB, it's essentially free — you can
# run it alongside any other model without VRAM pressure.
#
# See internal-docs/guides/lightweight-companion-models.md for the full
# rationale and task-routing guide (when to use lite vs primary).
#
# Auto-detects system and selects the appropriate model + runtime:
#
# NVIDIA (DGX Spark GB10):  Qwen3-4B-Instruct-AWQ-4bit  (SGLang)
#   - ~2.5 GB VRAM, ~150+ tok/s generation
#   - Same Qwen3 family as the 80B primary models (consistent tokenizer/behavior)
# Apple Silicon (Mac):      mlx-community Qwen3-4B-Instruct-4bit  (MLX)
#   - ~2.5 GB unified memory
#   - On 16GB Macs, this can serve as the ONLY model (fits with room to spare)
# Vulkan:                   Qwen3-4B-Instruct-GGUF  (vLLM)
#   - GGUF format for Vulkan/CPU fallback
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../libexec/runner-lib.sh
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/libexec/runner-lib.sh"

# Per-system model configuration
configure_model() {
  local system
  system="$(detect_system)"
  echo "Detected system: $system"

  case "$system" in
    nvidia)
      # Qwen3-4B-Instruct in AWQ-4bit: ~2.5 GB VRAM.
      # Uses SGLang (the default for dense models on GB10 per ADR-202608021744).
      # This model is tiny — GPU_MEMORY_UTILIZATION can be low (0.50) since
      # there's no need to reserve VRAM for a large KV cache. The remaining
      # VRAM is available for co-located models.
      ACCT="${ACCT:-Qwen}"
      MODEL="${MODEL:-Qwen3-4B-Instruct-AWQ}"
      HOST_PORT="${HOST_PORT:-8004}"
      RUNTIME="${RUNTIME:-sglang}"
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
      ;;
    apple-silicon)
      # MLX-converted Qwen3-4B-Instruct from mlx-community.
      # Pre-converted to MLX format — no auto-conversion needed (saves SSD
      # writes on Macs with soldered storage; see SSD Wear Mitigation guide).
      ACCT="${ACCT:-mlx-community}"
      MODEL="${MODEL:-Qwen3-4B-Instruct-4bit}"
      HOST_PORT="${HOST_PORT:-8004}"
      RUNTIME="${RUNTIME:-mlx}"
      ;;
    vulkan)
      # GGUF format for Vulkan/CPU fallback.
      # Q4_K_M quantization: ~2.5 GB, good quality.
      ACCT="${ACCT:-Qwen}"
      MODEL="${MODEL:-Qwen3-4B-Instruct-GGUF}"
      HOST_PORT="${HOST_PORT:-8004}"
      RUNTIME="${RUNTIME:-vllm}"
      ;;
    *)
      echo "Error: Unsupported system '$system' for lite model." >&2
      echo "Set SYSTEM=nvidia|apple-silicon|vulkan or ACCT/MODEL/RUNTIME manually." >&2
      exit 1
      ;;
  esac
}

main() {
  load_env
  configure_model
  check_prerequisites
  setup_huggingface
  clear_caches
  stop_existing_container "$MODEL"

  # The lite model doesn't need special flags — it's small and fast.
  # FP8 KV cache is optional here since the model is so small that KV cache
  # is rarely a constraint. Enable it with KV_CACHE_DTYPE=fp8 if you want
  # to maximize concurrent sessions on the lite endpoint.
  run_container "$ACCT" "$MODEL" "$HOST_PORT"
}

main "$@"
