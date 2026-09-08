#!/usr/bin/env bash
#
# TTS Model Runner — Text-to-Speech (Kokoro-82M)
#
# Serves a Kokoro-82M text-to-speech model with an OpenAI-compatible
# /v1/audio/speech endpoint. Kokoro is a lightweight TTS model (82M
# parameters, ~300 MB) that produces high-quality speech in 9 languages
# with 54+ voices.
#
# Why Kokoro:
#   - Extremely lightweight: ~0.3 GB VRAM (barely affects co-location budget)
#   - OpenAI-compatible API: /v1/audio/speech (any OpenAI TTS client works)
#   - 54+ voices across 9 languages (English, Japanese, Chinese, Spanish,
#     French, Italian, and more)
#   - All major audio formats: mp3, wav, flac, opus, aac, pcm
#   - Streaming support for low time-to-first-audio
#   - Accepts OpenAI voice-name aliases (alloy, nova, echo, ...) that map
#     to local Kokoro voices
#
# Docker image: hwdsl2/kokoro-server (OpenAI-compatible, multi-arch)
#   - :cuda tag for NVIDIA GPU acceleration
#   - default tag for CPU
#   - Persistent model cache via Docker volume
#   - Offline/air-gapped mode supported (KOKORO_LOCAL_ONLY=1)
#
# On Apple Silicon, Docker doesn't support GPU passthrough, so the script
# falls back to CPU inference (Kokoro is small enough for real-time CPU
# synthesis) or can use a native Python process with MPS acceleration.
#
# Port: 8005 (unique, allows co-location with LLM models)
# API:  POST /v1/audio/speech  (OpenAI-compatible)
#
# Usage:
#   ./model-tts.sh                    # start TTS server on port 8005
#   HOST_PORT=9005 ./model-tts.sh      # custom port
#   CUSTOM_IMAGE=hwdsl2/kokoro-server:cuda ./model-tts.sh  # override image
#
# Test:
#   curl -X POST http://localhost:8005/v1/audio/speech \
#     -H "Content-Type: application/json" \
#     -d '{"model":"kokoro","input":"Hello world","voice":"af_heart","format":"mp3"}' \
#     --output speech.mp3
#
# See:
#   - internal-docs/platforms/nvidia-gb10-128gb.md (co-location with LLMs)
#   - internal-docs/guides/ssd-wear-mitigation.md (model cache on external SSD)
#   - https://github.com/hwdsl2/docker-kokoro (Docker image source)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./runner-lib.sh
source "$SCRIPT_DIR/runner-lib.sh"

# Kokoro TTS Docker image (pinned to a specific tag for reproducibility).
# The :cuda tag includes CUDA support for NVIDIA GPU acceleration.
# The default (CPU) tag works on any platform.
# Multi-arch: linux/amd64, linux/arm64.
# See https://hub.docker.com/r/hwdsl2/kokoro-server for available tags.
KOKORO_IMAGE_CUDA="hwdsl2/kokoro-server:cuda"
KOKORO_IMAGE_CPU="hwdsl2/kokoro-server"

# Kokoro server's internal port is 8880 (not 8000 like vLLM/SGLang).
# The run_custom_container function maps host_port -> internal_port.
KOKORO_INTERNAL_PORT=8880

# Per-system configuration
configure_tts() {
  local system
  system="$(detect_system)"
  echo "Detected system: $system"

  case "$system" in
    nvidia)
      # NVIDIA GPU: use the :cuda image for hardware-accelerated TTS.
      # Kokoro-82M is tiny (~300 MB), so GPU inference is very fast.
      # The :cuda tag requires NVIDIA driver 575+ and NVIDIA Container Toolkit.
      TTS_IMAGE="${TTS_IMAGE:-$KOKORO_IMAGE_CUDA}"
      TTS_CONTAINER_NAME="${TTS_CONTAINER_NAME:-kokoro-tts}"
      HOST_PORT="${HOST_PORT:-8005}"
      CUSTOM_CONTAINER_PORT="$KOKORO_INTERNAL_PORT"
      USE_DOCKER=1
      ;;
    apple-silicon)
      # Apple Silicon: Docker on Mac doesn't support GPU passthrough.
      # Kokoro is small enough for real-time CPU inference.
      # Alternatively, use a native Python process with MPS (Metal) acceleration:
      #   pip install kokoro-onnx
      #   python -m kokoro.server --port 8005
      # For simplicity, we use the CPU Docker image here.
      TTS_IMAGE="${TTS_IMAGE:-$KOKORO_IMAGE_CPU}"
      TTS_CONTAINER_NAME="${TTS_CONTAINER_NAME:-kokoro-tts}"
      HOST_PORT="${HOST_PORT:-8005}"
      CUSTOM_CONTAINER_PORT="$KOKORO_INTERNAL_PORT"
      USE_DOCKER=1
      ;;
    vulkan|cpu)
      # Vulkan/CPU: use the CPU Docker image.
      # Kokoro doesn't use Vulkan — CPU inference is sufficient for TTS.
      TTS_IMAGE="${TTS_IMAGE:-$KOKORO_IMAGE_CPU}"
      TTS_CONTAINER_NAME="${TTS_CONTAINER_NAME:-kokoro-tts}"
      HOST_PORT="${HOST_PORT:-8005}"
      CUSTOM_CONTAINER_PORT="$KOKORO_INTERNAL_PORT"
      USE_DOCKER=1
      ;;
    *)
      echo "Error: Unsupported system '$system' for TTS." >&2
      echo "Set SYSTEM=nvidia|apple-silicon|vulkan|cpu or TTS_IMAGE manually." >&2
      exit 1
      ;;
  esac
}

main() {
  load_env
  configure_tts

  # check_prerequisites checks for docker/sudo based on RUNTIME, but TTS
  # doesn't use the LLM runtime dispatch. Check docker directly if using Docker.
  if [ "$USE_DOCKER" = "1" ]; then
    command -v docker >/dev/null 2>&1 || {
      echo "Error: docker not found. Install Docker or use a native TTS server." >&2
      exit 1
    }
    if ! docker info >/dev/null 2>&1; then
      echo "Error: Docker is not running. Please start Docker and try again." >&2
      exit 1
    fi
  fi

  # Clear caches (helps prevent OOM on the host when co-locating with LLMs).
  # On Macs this uses `purge` instead of /proc/sys/vm/drop_caches.
  clear_caches

  # Stop any existing TTS container/process.
  stop_existing_container "$TTS_CONTAINER_NAME"

  # Run the TTS container.
  # run_custom_container provides:
  #   --gpus all (NVIDIA only)
  #   --ipc=host
  #   --shm-size 8g
  #   HF cache volume (reuses downloaded models)
  #   HF token env vars
  #   Port mapping: host_port -> internal_port (8880)
  #
  # The Kokoro image has a default ENTRYPOINT that starts the server,
  # so we don't need to pass a command — the image handles it.
  #
  # Environment variables for the Kokoro container:
  #   KOKORO_LOCAL_ONLY=1  — air-gapped mode (use pre-cached model only)
  #   KOKORO_PORT=8880     — internal port (already set via CUSTOM_CONTAINER_PORT)
  #
  # To enable offline mode, set KOKORO_LOCAL_ONLY=1 in .env or export it.
  # This prevents the container from downloading the model on startup
  # (useful for air-gapped deployments or to avoid re-downloads).
  run_custom_container "$TTS_IMAGE" "$TTS_CONTAINER_NAME" "$HOST_PORT"
}

main "$@"
