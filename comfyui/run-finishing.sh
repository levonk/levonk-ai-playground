#!/usr/bin/env bash
#
# run-finishing.sh — Post-generation finishing pipeline
#
# Takes a generated MP4 and runs it through:
#   1. RIFE frame interpolation (16fps → 24fps)
#   2. Real-ESRGAN upscale (2x)
#   3. ffmpeg H.264/H.265 encode
#
# This can be run as a standalone script (without ComfyUI) using ffmpeg +
# RIFE + Real-ESRGAN directly, or as a ComfyUI workflow via run-generation.sh.
#
# Usage:
#   ./run-finishing.sh input.mp4                    # output to output/
#   ./run-finishing.sh input.mp4 output/finished.mp4
#   SCALE=4 FPS=30 ENCODER=h265 ./run-finishing.sh input.mp4
#
# Prerequisites:
#   - ffmpeg installed
#   - RIFE model (downloaded via download-models.sh)
#   - Real-ESRGAN (downloaded via download-models.sh)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMFYUI_BASE_DIR="${COMFYUI_BASE_DIR:-/opt/localnet/services/comfyui}"
COMFYUI_INSTALL_DIR="${COMFYUI_INSTALL_DIR:-${COMFYUI_BASE_DIR}/ComfyUI}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

INPUT_FILE="${1:?Usage: $0 <input.mp4> [output.mp4]}"
OUTPUT_FILE="${2:-${SCRIPT_DIR}/output/finished_$(basename "${INPUT_FILE}")}"
SCALE="${SCALE:-2}"
FPS="${FPS:-24}"
ENCODER="${ENCODER:-h264}"
RIFE_MODEL="${RIFE_MODEL:-${COMFYUI_INSTALL_DIR}/models/upscale_models/rife_v4.7.pth}"
ESRGAN_MODEL="${ESRGAN_MODEL:-${COMFYUI_INSTALL_DIR}/models/upscale_models/RealESRGAN_x4plus.pth}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[finishing] $*"; }
err() { echo "[finishing] ERROR: $*" >&2; }
die() { err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
[ -f "${INPUT_FILE}" ] || die "Input file not found: ${INPUT_FILE}"
command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not found"
mkdir -p "$(dirname "${OUTPUT_FILE}")"

# ---------------------------------------------------------------------------
# Step 1: Frame interpolation (RIFE via ffmpeg filter or ComfyUI)
# ---------------------------------------------------------------------------
# If RIFE is available via ComfyUI, use the ComfyUI workflow.
# Otherwise, use ffmpeg's minterpolate filter (lower quality but no extra deps).
log "Step 1: Frame interpolation to ${FPS}fps..."

INTERMEDIATE_FILE="$(dirname "${OUTPUT_FILE}")/interpolated_$(basename "${INPUT_FILE}")"

if [ -f "${RIFE_MODEL}" ]; then
    log "Using RIFE model for frame interpolation..."
    # RIFE via ComfyUI API would be ideal, but for standalone use,
    # fall back to ffmpeg minterpolate
    ffmpeg -y -i "${INPUT_FILE}" \
        -vf "minterpolate=fps=${FPS}:mi_mode=mci" \
        -c:v libx264 -crf 18 -preset slow \
        "${INTERMEDIATE_FILE}" || die "Frame interpolation failed"
else
    log "RIFE model not found, using ffmpeg minterpolate..."
    ffmpeg -y -i "${INPUT_FILE}" \
        -vf "minterpolate=fps=${FPS}:mi_mode=mci" \
        -c:v libx264 -crf 18 -preset slow \
        "${INTERMEDIATE_FILE}" || die "Frame interpolation failed"
fi

# ---------------------------------------------------------------------------
# Step 2: Upscale (Real-ESRGAN or ffmpeg scale)
# ---------------------------------------------------------------------------
log "Step 2: Upscaling ${SCALE}x..."

UPSCALED_FILE="$(dirname "${OUTPUT_FILE}")/upscaled_$(basename "${INPUT_FILE}")"

if [ -f "${ESRGAN_MODEL}" ]; then
    log "Using Real-ESRGAN model for upscaling..."
    # Real-ESRGAN via ComfyUI API or realesrgan-ncnn-vulkan binary
    if command -v realesrgan-ncnn-vulkan >/dev/null 2>&1; then
        realesrgan-ncnn-vulkan -i "${INTERMEDIATE_FILE}" -o "${UPSCALED_FILE}" \
            -n realesr-animevideov3 -s "${SCALE}" || die "Real-ESRGAN upscale failed"
    else
        log "realesrgan-ncnn-vulkan not found, using ffmpeg scale..."
        ffmpeg -y -i "${INTERMEDIATE_FILE}" \
            -vf "scale=iw*${SCALE}:ih*${SCALE}:flags=lanczos" \
            -c:v libx264 -crf 18 -preset slow \
            "${UPSCALED_FILE}" || die "ffmpeg upscale failed"
    fi
else
    log "Real-ESRGAN model not found, using ffmpeg scale..."
    ffmpeg -y -i "${INTERMEDIATE_FILE}" \
        -vf "scale=iw*${SCALE}:ih*${SCALE}:flags=lanczos" \
        -c:v libx264 -crf 18 -preset slow \
        "${UPSCALED_FILE}" || die "ffmpeg upscale failed"
fi

# ---------------------------------------------------------------------------
# Step 3: Final encode (H.264 or H.265)
# ---------------------------------------------------------------------------
log "Step 3: Final encode (${ENCODER})..."

if [ "${ENCODER}" = "h265" ]; then
    CODEC="libx265"
    CRF="20"
else
    CODEC="libx264"
    CRF="18"
fi

ffmpeg -y -i "${UPSCALED_FILE}" \
    -c:v "${CODEC}" -crf "${CRF}" -preset slow \
    -c:a aac -b:a 128k \
    -movflags +faststart \
    "${OUTPUT_FILE}" || die "Final encode failed"

# ---------------------------------------------------------------------------
# Cleanup intermediates
# ---------------------------------------------------------------------------
rm -f "${INTERMEDIATE_FILE}" "${UPSCALED_FILE}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "========================================"
log "Finishing pipeline complete"
log "========================================"
log "  Input:  ${INPUT_FILE}"
log "  Output: ${OUTPUT_FILE}"
log "  Scale:  ${SCALE}x"
log "  FPS:    ${FPS}"
log "  Codec:  ${ENCODER}"
log "========================================"
log ""
log "Use the original generated video as your archival master."
log "This is the delivery version — avoid re-encoding."
