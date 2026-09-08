#!/usr/bin/env bash
#
# run-generation.sh — Queue a ComfyUI workflow via the API and save the output
#
# Loads a workflow JSON, submits it to the ComfyUI API, waits for completion,
# and downloads the output video. This is the "smoke-test" equivalent of
# test-query.sh for the video generation pipeline.
#
# Usage:
#   ./run-generation.sh                                    # default workflow
#   ./run-generation.sh workflows/wan-i2v-generation.json  # specific workflow
#   COMFYUI_HOST=ai COMFYUI_PORT=8188 ./run-generation.sh   # remote host
#
# Prerequisites:
#   - ComfyUI running (systemd service or manual)
#   - Model checkpoints downloaded (download-models.sh)
#   - Reference image + keyframe in ComfyUI/input/

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMFYUI_HOST="${COMFYUI_HOST:-localhost}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"
COMFYUI_URL="http://${COMFYUI_HOST}:${COMFYUI_PORT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKFLOW_FILE="${1:-${SCRIPT_DIR}/workflows/wan-i2v-generation.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[comfyui-gen] $*"; }
err() { echo "[comfyui-gen] ERROR: $*" >&2; }
die() { err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
[ -f "${WORKFLOW_FILE}" ] || die "Workflow file not found: ${WORKFLOW_FILE}"
mkdir -p "${OUTPUT_DIR}"

# Check ComfyUI is running
log "Checking ComfyUI at ${COMFYUI_URL}..."
if ! curl -sf "${COMFYUI_URL}/system_stats" >/dev/null 2>&1; then
    die "ComfyUI not reachable at ${COMFYUI_URL}. Start it with: sudo systemctl start comfyui"
fi
log "ComfyUI is running."

# ---------------------------------------------------------------------------
# Load and submit workflow
# ---------------------------------------------------------------------------
log "Loading workflow: ${WORKFLOW_FILE}"

# ComfyUI API expects the workflow in a specific format.
# The workflow JSON is the API-format prompt (not the UI-save format).
# For UI-saved workflows, you may need to convert via ComfyUI-Manager or
# the /api/prompt endpoint with the "prompt" field.

# Read the workflow JSON
WORKFLOW_JSON=$(cat "${WORKFLOW_FILE}")

# Submit the prompt
log "Submitting prompt to ComfyUI API..."
PROMPT_RESPONSE=$(curl -sf -X POST "${COMFYUI_URL}/prompt" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": ${WORKFLOW_JSON}}") || die "Failed to submit prompt"

# Extract prompt ID
PROMPT_ID=$(echo "${PROMPT_RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['prompt_id'])" 2>/dev/null || echo "")

if [ -z "${PROMPT_ID}" ]; then
    die "Failed to get prompt ID from response: ${PROMPT_RESPONSE}"
fi

log "Prompt ID: ${PROMPT_ID}"

# ---------------------------------------------------------------------------
# Wait for completion
# ---------------------------------------------------------------------------
log "Waiting for generation to complete..."
TIMEOUT=600  # 10 minutes max
ELAPSED=0

while [ ${ELAPSED} -lt ${TIMEOUT} ]; do
    HISTORY=$(curl -sf "${COMFYUI_URL}/history/${PROMPT_ID}" 2>/dev/null || echo "")

    if echo "${HISTORY}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if '${PROMPT_ID}' in data:
    status = data['${PROMPT_ID}'].get('status', {})
    if status.get('completed', False):
        sys.exit(0)
    if status.get('status_str') == 'error':
        sys.exit(1)
sys.exit(2)
" 2>/dev/null; then
        log "Generation complete!"
        break
    elif [ $? -eq 1 ]; then
        die "Generation failed (error status)"
    fi

    sleep 10
    ELAPSED=$((ELAPSED + 10))
    log "Waiting... (${ELAPSED}s elapsed)"
done

if [ ${ELAPSED} -ge ${TIMEOUT} ]; then
    die "Generation timed out after ${TIMEOUT}s"
fi

# ---------------------------------------------------------------------------
# Download output
# ---------------------------------------------------------------------------
log "Fetching output files..."
HISTORY=$(curl -sf "${COMFYUI_URL}/history/${PROMPT_ID}")

# Extract output filenames and download them
echo "${HISTORY}" | python3 -c "
import sys, json, urllib.request, os

data = json.load(sys.stdin)
prompt_data = data.get('${PROMPT_ID}', {})
outputs = prompt_data.get('outputs', {})

base_url = '${COMFYUI_URL}'
output_dir = '${OUTPUT_DIR}'

for node_id, node_output in outputs.items():
    # Video outputs
    for gif in node_output.get('gifs', []):
        fname = gif.get('filename', f'output_{node_id}.mp4')
        subfolder = gif.get('subfolder', '')
        url = f'{base_url}/view?filename={fname}&subfolder={subfolder}&type=output'
        dest = os.path.join(output_dir, fname)
        print(f'Downloading {fname}...')
        urllib.request.urlretrieve(url, dest)
        print(f'Saved to {dest}')

    # Image outputs
    for img in node_output.get('images', []):
        fname = img.get('filename', f'output_{node_id}.png')
        subfolder = img.get('subfolder', '')
        url = f'{base_url}/view?filename={fname}&subfolder={subfolder}&type=output'
        dest = os.path.join(output_dir, fname)
        print(f'Downloading {fname}...')
        urllib.request.urlretrieve(url, dest)
        print(f'Saved to {dest}')
" || die "Failed to download outputs"

log "========================================"
log "Generation complete"
log "========================================"
log "  Output dir: ${OUTPUT_DIR}"
log "  Prompt ID: ${PROMPT_ID}"
log "========================================"
