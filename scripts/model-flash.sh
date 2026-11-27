#!/usr/bin/env bash
#
# Flash Model Runner — Qwen3.8-Flash-Next (frontier MoE, multimodal)
# Auto-detects system and selects the appropriate model + runtime.
#
# NVIDIA (DGX Spark GB10):  Qwen3.8-Flash-Next NVFP4 pack  (Strata)
#   - Strata ships a linux-aarch64 engine with sm_121-tuned defaults and
#     --mmap-experts for the 128 GB unified memory pool.
#   - Upstream measured on a Spark: ~43-46 tok/s decode, ~1,400-1,700 tok/s
#     prompt (64K context). This is the fastest engine for this model on GB10.
#   - Default pack: NVFP4 (~68 GB experts, best quality).
#     Lighter/faster alternative: STRATA_FAMILY=qwen STRATA_PACK=iq3_s
#     (~50 GB experts, ~46 tok/s decode).
# Apple Silicon (Mac):      mlx-community Qwen3.8-Flash-Next-4bit  (MLX)
#   - ~112 GB unified memory — only fits 128 GB Macs.
#
# Basalt (the other Blackwell engine) is NOT an option on the Spark:
# it builds x86-64/sm_120 only. See internal-docs/guides/runtime-selection.md.
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
      # Strata catalog pack: the engine downloads and tunes it itself.
      ACCT="${ACCT:-nvidia}"
      MODEL="${MODEL:-Qwen3.8-Flash-Next}"
      HOST_PORT="${HOST_PORT:-8007}"
      RUNTIME="${RUNTIME:-strata}"
      STRATA_FAMILY="${STRATA_FAMILY:-nvfp4}"
      STRATA_PACK="${STRATA_PACK:-nvfp4}"
      ;;
    apple-silicon)
      ACCT="${ACCT:-mlx-community}"
      MODEL="${MODEL:-Qwen3.8-Flash-Next-4bit}"
      HOST_PORT="${HOST_PORT:-8007}"
      RUNTIME="${RUNTIME:-mlx}"
      ;;
    *)
      echo "Error: Unsupported system '$system' for Qwen3.8-Flash-Next." >&2
      echo "Set SYSTEM=nvidia|apple-silicon or ACCT/MODEL/RUNTIME manually." >&2
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
