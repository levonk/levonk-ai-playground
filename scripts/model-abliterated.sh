#!/usr/bin/env bash
#
# Abliterated Model Runner — dedicated uncensored endpoint
#
# Abliteration (refusal-direction removal) is a VARIANT of a role, not a new
# role. This script is the single abliterated endpoint — it does not need a
# per-category twin because you would not run an aligned and an abliterated
# copy of the same 80B-class model simultaneously (they would not both fit
# anyway). If an abliterated model proves better than an aligned one for a
# category, swap that category's MODEL instead of adding another script.
#
# Why Qwen3.8-27B bf16 as the default: ~54 GB of weights leaves ~70 GB of
# the GB10's unified memory free, so the abliterated endpoint can co-run
# with another large model. It is also a generation newer than the
# Qwen3-Next-80B models in the other categories.
#
# NVIDIA (DGX Spark GB10):  huihui-ai Huihui-Qwen3.8-27B-abliterated (SGLang, bf16)
#   - ~54 GB download: co-location-friendly default.
#   - vLLM fallback if SGLang rejects the arch on this image:
#       RUNTIME=vllm ./scripts/model-abliterated.sh
#   - Max-quality alternative (fills the box, ~84 GB):
#       RUNTIME=strata \
#       MODEL=Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF \
#       STRATA_GGUF_PATTERN='swift-IQ3_S/*' \
#       ./scripts/model-abliterated.sh
#   - Max-quality vLLM alternative (~100 GB):
#       RUNTIME=vllm ACCT=Jiunsong \
#       MODEL=SuperQwen3.8-Flash-Next-abliterated-FP8-DGX-Spark \
#       ./scripts/model-abliterated.sh
# Apple Silicon (Mac):      mlx-community Claude-4.6-Opus-abliterated-4bit (MLX)
#   - ~20 GB — the practical abliterated pick for unified-memory Macs.
# Vulkan:                   mradermacher 80B-A3B-Instruct-abliterated GGUF (vLLM)
#
# Picked via `llmfit --profile gb10-spark` (128 GB unified, 273 GB/s):
# top abliterated fits — Huihui-Qwen3.8-Flash-Next-abliterated (~87),
# Huihui-Qwen3-Coder-Next-abliterated (~88), Huihui-Qwen3.6-35B-A3B-abliterated
# (~88), Huihui-Qwen3-Next-80B-A3B-Thinking-abliterated (~85). Direct
# abliterated twins also exist for the code/reason/lite categories — see
# internal-docs/guides/runtime-selection.md for the mapping table.
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
      ACCT="${ACCT:-huihui-ai}"
      MODEL="${MODEL:-Huihui-Qwen3.8-27B-abliterated}"
      HOST_PORT="${HOST_PORT:-8008}"
      RUNTIME="${RUNTIME:-sglang}"
      ;;
    apple-silicon)
      ACCT="${ACCT:-mlx-community}"
      MODEL="${MODEL:-Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-4bit}"
      HOST_PORT="${HOST_PORT:-8008}"
      RUNTIME="${RUNTIME:-mlx}"
      ;;
    vulkan)
      ACCT="${ACCT:-mradermacher}"
      MODEL="${MODEL:-Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-GGUF}"
      HOST_PORT="${HOST_PORT:-8008}"
      RUNTIME="${RUNTIME:-vllm}"
      ;;
    *)
      echo "Error: Unsupported system '$system' for abliterated model." >&2
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

  run_container "$ACCT" "$MODEL" "$HOST_PORT"
}

main "$@"
