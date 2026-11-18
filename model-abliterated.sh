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
# Why Flash-Next as the default: it is a generation newer than the
# Qwen3-Next-80B models in the other categories, and under Strata on the GB10
# it is also the fastest option (~46 tok/s decode upstream-measured) despite
# being the biggest model — the trade-off is memory (~84 GB), which leaves
# ~35 GB and blocks co-running another large model.
#
# NVIDIA (DGX Spark GB10):  huihui-ai Qwen3.8-Flash-Next-abliterated-v2-GGUF
#   - Strata + swift-IQ3_S pack (~84 GB download): best quality+speed combo.
#   - Co-location-friendly alternative (leaves ~75 GB):
#       RUNTIME=vllm ACCT=coughmedicine \
#       MODEL=Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-W4A16 \
#       ./model-abliterated.sh
#   - Max-quality alternative (fills the box, ~100 GB):
#       RUNTIME=vllm ACCT=Jiunsong \
#       MODEL=SuperQwen3.8-Flash-Next-abliterated-FP8-DGX-Spark \
#       ./model-abliterated.sh
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
# shellcheck source=./runner-lib.sh
source "$SCRIPT_DIR/runner-lib.sh"

# Per-system model configuration
configure_model() {
  local system
  system="$(detect_system)"
  echo "Detected system: $system"

  case "$system" in
    nvidia)
      ACCT="${ACCT:-huihui-ai}"
      MODEL="${MODEL:-Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF}"
      HOST_PORT="${HOST_PORT:-8008}"
      RUNTIME="${RUNTIME:-strata}"
      # swift-IQ3_S shards only (~84 GB of the ~1 TB quant zoo in this repo)
      STRATA_GGUF_PATTERN="${STRATA_GGUF_PATTERN:-swift-IQ3_S/*}"
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
