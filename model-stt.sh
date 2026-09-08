#!/usr/bin/env bash
#
# STT Model Runner — Speech-to-Text (Whisper large-v3-turbo)
#
# Serves a Whisper speech-to-text model with an OpenAI-compatible
# /v1/audio/transcriptions endpoint. Powered by faster-whisper
# (CTranslate2 reimplementation — ~4x faster, half the VRAM of the
# reference OpenAI Whisper implementation).
#
# Why Whisper large-v3-turbo:
#   - Best quality open-weights STT model (multilingual)
#   - faster-whisper backend: INT8 quantization, ~1.5 GB VRAM
#   - OpenAI-compatible API: /v1/audio/transcriptions
#   - Streaming support (SSE) for real-time transcription
#   - Translation support: /v1/audio/translations
#   - JSON, SRT, VTT, and text output formats
#
# Docker image: hwdsl2/whisper-server (OpenAI-compatible, multi-arch)
#   - :cuda tag for NVIDIA GPU acceleration (linux/amd64 only)
#   - default tag for CPU (linux/amd64 + linux/arm64)
#   - Persistent model cache via Docker volume
#   - Offline/air-gapped mode supported (WHISPER_LOCAL_ONLY=1)
#
# On Apple Silicon, Docker doesn't support GPU passthrough, so the script
# uses the CPU Docker image (Whisper turbo is fast enough on CPU for
# real-time transcription) or can use a native Python process with MPS.
#
# Port: 8006 (unique, allows co-location with LLM models)
# API:  POST /v1/audio/transcriptions  (OpenAI-compatible)
#       POST /v1/audio/translations     (OpenAI-compatible)
#
# Usage:
#   ./model-stt.sh                    # start STT server on port 8006
#   HOST_PORT=9006 ./model-stt.sh      # custom port
#   CUSTOM_IMAGE=hwdsl2/whisper-server:cuda ./model-stt.sh  # override image
#
# Test:
#   curl -X POST http://localhost:8006/v1/audio/transcriptions \
#     -H "Content-Type: multipart/form-data" \
#     -F file=@audio.mp3 \
#     -F model=whisper-large-v3-turbo \
#     -F response_format=json
#
# Environment variables for the Whisper container:
#   WHISPER_MODEL         — model name (default: large-v3-turbo)
#   WHISPER_DEVICE        — cpu, cuda, or auto (default: auto)
#   WHISPER_LOCAL_ONLY    — 1 for air-gapped mode (pre-cached model only)
#   WHISPER_LANGUAGE      — language code (e.g. "en", "ja", "auto")
#
# See:
#   - internal-docs/platforms/nvidia-gb10-128gb.md (co-location with LLMs)
#   - internal-docs/guides/ssd-wear-mitigation.md (model cache on external SSD)
#   - https://github.com/hwdsl2/docker-whisper (Docker image source)
#   - https://github.com/SYSTRAN/faster-whisper (faster-whisper backend)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./runner-lib.sh
source "$SCRIPT_DIR/runner-lib.sh"

# Whisper STT Docker image (pinned for reproducibility).
# The :cuda tag includes CUDA support for NVIDIA GPU acceleration.
# The default (CPU) tag works on any platform (amd64 + arm64).
# See https://hub.docker.com/r/hwdsl2/whisper-server for available tags.
WHISPER_IMAGE_CUDA="hwdsl2/whisper-server:cuda"
WHISPER_IMAGE_CPU="hwdsl2/whisper-server"

# Whisper server's internal port is 9000 (not 8000 like vLLM/SGLang).
# The run_custom_container function maps host_port -> internal_port.
WHISPER_INTERNAL_PORT=9000

# Per-system configuration
configure_stt() {
  local system
  system="$(detect_system)"
  echo "Detected system: $system"

  case "$system" in
    nvidia)
      # NVIDIA GPU: use the :cuda image for hardware-accelerated STT.
      # faster-whisper with INT8 quantization uses ~1.5 GB VRAM for
      # large-v3-turbo — negligible compared to LLM models.
      # The :cuda tag requires NVIDIA driver 575+ and NVIDIA Container Toolkit.
      # Note: :cuda image is linux/amd64 only (no arm64 CUDA build).
      STT_IMAGE="${STT_IMAGE:-$WHISPER_IMAGE_CUDA}"
      STT_CONTAINER_NAME="${STT_CONTAINER_NAME:-whisper-stt}"
      HOST_PORT="${HOST_PORT:-8006}"
      CUSTOM_CONTAINER_PORT="$WHISPER_INTERNAL_PORT"
      USE_DOCKER=1
      ;;
    apple-silicon)
      # Apple Silicon: Docker on Mac doesn't support GPU passthrough.
      # Use the CPU image — Whisper turbo is fast enough on Apple Silicon
      # CPU for real-time transcription (the M-series CPU is very capable).
      # Alternatively, use a native Python process with MPS acceleration:
      #   pip install faster-whisper
      #   python -c "from faster_whisper import WhisperModel; ..."
      # Or use mlx-whisper for native MLX acceleration:
      #   pip install mlx-whisper
      STT_IMAGE="${STT_IMAGE:-$WHISPER_IMAGE_CPU}"
      STT_CONTAINER_NAME="${STT_CONTAINER_NAME:-whisper-stt}"
      HOST_PORT="${HOST_PORT:-8006}"
      CUSTOM_CONTAINER_PORT="$WHISPER_INTERNAL_PORT"
      USE_DOCKER=1
      ;;
    vulkan|cpu)
      # Vulkan/CPU: use the CPU Docker image.
      # Whisper doesn't use Vulkan — CPU inference works well for STT.
      STT_IMAGE="${STT_IMAGE:-$WHISPER_IMAGE_CPU}"
      STT_CONTAINER_NAME="${STT_CONTAINER_NAME:-whisper-stt}"
      HOST_PORT="${HOST_PORT:-8006}"
      CUSTOM_CONTAINER_PORT="$WHISPER_INTERNAL_PORT"
      USE_DOCKER=1
      ;;
    *)
      echo "Error: Unsupported system '$system' for STT." >&2
      echo "Set SYSTEM=nvidia|apple-silicon|vulkan|cpu or STT_IMAGE manually." >&2
      exit 1
      ;;
  esac
}

main() {
  load_env
  configure_stt

  # Check Docker availability (TTS/STT don't use the LLM runtime dispatch,
  # so check_prerequisites() isn't appropriate here).
  if [ "$USE_DOCKER" = "1" ]; then
    command -v docker >/dev/null 2>&1 || {
      echo "Error: docker not found. Install Docker or use a native STT server." >&2
      exit 1
    }
    if ! docker info >/dev/null 2>&1; then
      echo "Error: Docker is not running. Please start Docker and try again." >&2
      exit 1
    fi
  fi

  # Clear caches (helps when co-locating with LLM models).
  clear_caches

  # Stop any existing STT container/process.
  stop_existing_container "$STT_CONTAINER_NAME"

  # Run the STT container.
  # run_custom_container provides:
  #   --gpus all (NVIDIA only)
  #   --ipc=host
  #   --shm-size 8g
  #   HF cache volume (reuses downloaded models)
  #   HF token env vars
  #   Port mapping: host_port -> internal_port (9000)
  #
  # The Whisper image has a default ENTRYPOINT that starts the server,
  # so we don't need to pass a command.
  #
  # To configure the Whisper model, set environment variables in .env:
  #   WHISPER_MODEL=large-v3-turbo   (default; alternatives: base, small,
  #                                   medium, large-v3, large-v3-turbo)
  #   WHISPER_DEVICE=auto            (auto, cpu, cuda)
  #   WHISPER_LOCAL_ONLY=1           (air-gapped mode — use pre-cached model)
  #   WHISPER_LANGUAGE=auto          (auto-detect, or "en", "ja", etc.)
  #
  # These are passed to the container as environment variables by the
  # Docker image's entrypoint. The image reads them at startup.
  #
  # Note: The run_custom_container function passes HUGGING_FACE_HUB_TOKEN
  # and HF_TOKEN to the container, so private/gated models work if you
  # have access.
  run_custom_container "$STT_IMAGE" "$STT_CONTAINER_NAME" "$HOST_PORT"
}

main "$@"
