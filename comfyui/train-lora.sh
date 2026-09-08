#!/usr/bin/env bash
#
# train-lora.sh — Train a character identity LoRA from a reference image pack
#
# Trains a lightweight LoRA for character identity preservation across
# keyframes and video generations. Uses ComfyUI's LoRA training nodes or
# a standalone training script (kohya-ss / sd-scripts).
#
# Scope (per OOS exception 2026-09-08):
#   - Lightweight single-GPU LoRA training for character identity ONLY
#   - 20-40 image reference pack
#   - NOT for full fine-tuning, quantization, or pre-training
#
# Usage:
#   ./train-lora.sh                                    # train from default pack
#   ./train-lora.sh reference-packs/zvra/              # specific pack
#   ./train-lora.sh reference-packs/zvra/ --steps 2000 # custom steps
#
# Prerequisites:
#   - ComfyUI installed (install-comfyui.sh)
#   - Reference image pack prepared (see reference-packs/README.md)
#   - SDXL/Flux base model downloaded

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMFYUI_BASE_DIR="${COMFYUI_BASE_DIR:-/opt/localnet/services/comfyui}"
COMFYUI_INSTALL_DIR="${COMFYUI_INSTALL_DIR:-${COMFYUI_BASE_DIR}/ComfyUI}"
COMFYUI_VENV_DIR="${COMFYUI_VENV_DIR:-${COMFYUI_BASE_DIR}/venv}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

REFERENCE_PACK="${1:-${SCRIPT_DIR}/reference-packs/default}"
LORA_OUTPUT_DIR="${LORA_OUTPUT_DIR:-${COMFYUI_INSTALL_DIR}/models/loras}"
BASE_MODEL="${BASE_MODEL:-${COMFYUI_INSTALL_DIR}/models/checkpoints/sd_xl_base_1.0.safetensors}"

# Training hyperparameters (conservative defaults per pipeline doc)
TRAIN_STEPS="${TRAIN_STEPS:-1500}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_DIM="${LORA_DIM:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESOLUTION="${RESOLUTION:-768}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-500}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[lora-train] $*"; }
err() { echo "[lora-train] ERROR: $*" >&2; }
die() { err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Parse extra args
# ---------------------------------------------------------------------------
shift || true
while [ $# -gt 0 ]; do
    case "$1" in
        --steps) TRAIN_STEPS="$2"; shift 2 ;;
        --lr) LEARNING_RATE="$2"; shift 2 ;;
        --rank) LORA_RANK="$2"; shift 2 ;;
        --batch) BATCH_SIZE="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
[ -d "${REFERENCE_PACK}" ] || die "Reference pack not found: ${REFERENCE_PACK}. See reference-packs/README.md"
[ -f "${BASE_MODEL}" ] || die "Base model not found: ${BASE_MODEL}. Run download-models.sh first."
[ -d "${COMFYUI_VENV_DIR}" ] || die "Venv not found. Run install-comfyui.sh first."

# Count images in reference pack
IMAGE_COUNT=$(find "${REFERENCE_PACK}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.webp" \) | wc -l | tr -d ' ')

if [ "${IMAGE_COUNT}" -lt 10 ]; then
    die "Only ${IMAGE_COUNT} images in reference pack. Need at least 20 (ideally 20-40)."
fi

if [ "${IMAGE_COUNT}" -lt 20 ]; then
    log "WARNING: Only ${IMAGE_COUNT} images. Recommended: 20-40. Quality may suffer."
fi

log "Activating venv..."
source "${COMFYUI_VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# Prepare training
# ---------------------------------------------------------------------------
CHARACTER_NAME=$(basename "${REFERENCE_PACK}")
LORA_NAME="${CHARACTER_NAME}_lora"
LORA_OUTPUT="${LORA_OUTPUT_DIR}/${LORA_NAME}.safetensors"

log "========================================"
log "LoRA Training"
log "========================================"
log "  Character:      ${CHARACTER_NAME}"
log "  Reference pack: ${REFERENCE_PACK}"
log "  Images:         ${IMAGE_COUNT}"
log "  Base model:     ${BASE_MODEL}"
log "  Output:         ${LORA_OUTPUT}"
log "  Steps:          ${TRAIN_STEPS}"
log "  Learning rate:  ${LEARNING_RATE}"
log "  LoRA rank:      ${LORA_RANK}"
log "  Batch size:     ${BATCH_SIZE}"
log "  Resolution:     ${RESOLUTION}"
log "========================================"

mkdir -p "${LORA_OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Train via kohya-ss / sd-scripts (if available) or ComfyUI LoRA Trainer
# ---------------------------------------------------------------------------
# Check if kohya-ss is available
if command -v kohya_ss >/dev/null 2>&1 || [ -d "${COMFYUI_BASE_DIR}/kohya_ss" ]; then
    log "Using kohya-ss for LoRA training..."

    KOHYA_DIR="${COMFYUI_BASE_DIR}/kohya_ss"

    # Prepare training config
    CONFIG_FILE="${COMFYUI_BASE_DIR}/training_${CHARACTER_NAME}.toml"
    cat > "${CONFIG_FILE}" <<TOML
[general]
enable_bucket = true

[model]
type = "sdxl"
base = "${BASE_MODEL}"

[datasets]
[[datasets.subsets]]
image_dir = "${REFERENCE_PACK}"
num_repeats = 3

[training]
output_dir = "${LORA_OUTPUT_DIR}"
output_name = "${LORA_NAME}"
save_every_n_steps = ${SAVE_EVERY_N_STEPS}
max_train_steps = ${TRAIN_STEPS}
learning_rate = ${LEARNING_RATE}
resolution = ${RESOLUTION}
batch_size = ${BATCH_SIZE}
lora_rank = ${LORA_RANK}
save_precision = "fp16"
TOML

    log "Training config: ${CONFIG_FILE}"

    # Run training
    cd "${KOHYA_DIR}"
    python3 train_network.py --config_file "${CONFIG_FILE}" || die "LoRA training failed"

elif [ -d "${COMFYUI_INSTALL_DIR}/custom_nodes/ComfyUI-Lora-Trainer" ]; then
    log "Using ComfyUI-Lora-Trainer for LoRA training..."

    # ComfyUI LoRA Trainer uses the ComfyUI API
    # Submit a training prompt via the API
    python3 - <<PY || die "LoRA training failed"
import json, urllib.request, time, sys

comfyui_url = "http://localhost:8188"
reference_pack = "${REFERENCE_PACK}"
base_model = "${BASE_MODEL}"
output_name = "${LORA_NAME}"
output_dir = "${LORA_OUTPUT_DIR}"
steps = ${TRAIN_STEPS}
lr = "${LEARNING_RATE}"
rank = ${LORA_RANK}
batch_size = ${BATCH_SIZE}
resolution = ${RESOLUTION}

# Submit training prompt
training_prompt = {
    "3": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": base_model.split("/")[-1]}
    },
    "train": {
        "class_type": "LoraTrainer",
        "inputs": {
            "model": ["3", 0],
            "image_dir": reference_pack,
            "output_name": output_name,
            "output_dir": output_dir,
            "steps": steps,
            "learning_rate": lr,
            "rank": rank,
            "batch_size": batch_size,
            "resolution": resolution
        }
    }
}

resp = urllib.request.urlopen(
    urllib.request.Request(
        f"{comfyui_url}/prompt",
        data=json.dumps({"prompt": training_prompt}).encode(),
        headers={"Content-Type": "application/json"}
    )
)
result = json.loads(resp.read())
prompt_id = result.get("prompt_id", "")
print(f"Training started, prompt_id: {prompt_id}")

# Wait for completion
timeout = 3600
elapsed = 0
while elapsed < timeout:
    try:
        hist = urllib.request.urlopen(f"{comfyui_url}/history/{prompt_id}")
        data = json.loads(hist.read())
        if prompt_id in data:
            status = data[prompt_id].get("status", {})
            if status.get("completed"):
                print("Training complete!")
                sys.exit(0)
            if status.get("status_str") == "error":
                print(f"Training error: {status}")
                sys.exit(1)
    except Exception:
        pass
    time.sleep(15)
    elapsed += 15
    if elapsed % 60 == 0:
        print(f"Waiting... ({elapsed}s)")

print("Training timed out")
sys.exit(1)
PY

else
    die "No LoRA training backend found. Install kohya-ss or ComfyUI-Lora-Trainer."
fi

# ---------------------------------------------------------------------------
# Verify output
# ---------------------------------------------------------------------------
if [ -f "${LORA_OUTPUT}" ]; then
    log "========================================"
    log "LoRA training complete"
    log "========================================"
    log "  Output: ${LORA_OUTPUT}"
    log "  Size: $(du -h "${LORA_OUTPUT}" | cut -f1)"
    log "========================================"
    log ""
    log "Next steps:"
    log "  1. Load the LoRA in ComfyUI (LoraLoader node)"
    log "  2. Set strength_model to 0.6-0.9"
    log "  3. Generate keyframes and verify identity retention"
    log "  4. If identity is weak, retrain with more steps or higher rank"
    log "========================================"
else
    die "LoRA output file not found: ${LORA_OUTPUT}"
fi
