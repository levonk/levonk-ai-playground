# ComfyUI — Image/Video Generation Pipeline

GPU workload scripts for ComfyUI running natively on the GB10 inference host
(kckinai). The host-side provisioning (ComfyUI install, PyTorch, custom nodes,
model downloads, systemd service, firewall, Traefik routing, DNS) is handled
by the infrahub project's `ai-comfyui` Ansible role.

This directory contains only the **GPU workload** scripts — generation,
finishing, and LoRA training — plus workflow definitions and reference image
packs.

## What lives here (playground = GPU workload)

- **run-generation.sh** — Queue a workflow via the ComfyUI API and save output
- **run-finishing.sh** — Post-generation: frame interpolation + upscale + encode
- **train-lora.sh** — Train a character identity LoRA from a reference pack
- **workflows/** — ComfyUI workflow JSON files (generation + finishing graphs)
- **reference-packs/** — Character reference image packs for LoRA training

## What lives in infrahub (host provisioning)

The `ai-comfyui` Ansible role handles:
- ComfyUI installation (Python venv, PyTorch ARM/Blackwell, git clone)
- Custom node installation (Wan, IP-Adapter, RIFE, Real-ESRGAN, etc.)
- Model checkpoint downloads
- Storage directories (models, LoRAs, output, input, custom_nodes, config)
- Systemd service (`comfyui.service`)
- Firewall rules
- Traefik routing + DNS (`comfyui.levonk.com`)

Deploy the host side from infrahub:
```bash
ansible-playbook \
  -i levonk/active/02-config/ansible/inventories/localnet.yml \
  shared/active/02-config/ansible/playbooks/deploy-comfyui.yml \
  --vault-password-file ~/.ansible/vault_password \
  --limit kckinai
```

## Pipeline

```
Character references
    ↓
Character LoRA / reference conditioning
    ↓
Still-image keyframes (SDXL / Flux)
    ↓
Wan 2.2 I2V animation
    ↓
Temporal consistency cleanup
    ↓
Upscale / frame interpolation (RIFE + Real-ESRGAN)
    ↓
Edit shots together with ffmpeg
```

## Quick Start

```bash
# 0. Deploy host via infrahub (installs ComfyUI, nodes, models, systemd)

# 1. Prepare a reference image pack (see reference-packs/README.md)

# 2. Train character LoRA
./train-lora.sh reference-packs/your-character/

# 3. Generate a video clip
./run-generation.sh workflows/wan-i2v-generation.json

# 4. Finish (interpolate + upscale + encode)
./run-finishing.sh output/wan_i2v_output_00001.mp4
```

## Suggested Starting Settings

| Setting | Starting point |
|---------|---------------|
| Resolution | 640x640, 832x480, or 480x832 |
| Frames | 49 or 81 |
| Clip duration | ~2-5 seconds |
| Batch size | 1 |
| Precision | BF16 or FP16 |
| LoRA strength | 0.6-0.9 |
| Motion strength | Moderate |
| Seed | Fixed while comparing prompts |

## First Experiment

1. Wan 2.2 14B I2V, 640x640, 49 frames
2. One character reference image + one character LoRA
3. Fixed seed, moderate motion
4. Three different actions:
   - Standing and looking at the camera
   - Walking toward the camera
   - Sitting down and turning

Once those work, increase to 81 frames, switch to widescreen, and add
pose/depth control.

## OOS Exception

LoRA training for character identity is in-scope per the OOS exception
(2026-09-08). Full fine-tuning, quantization, and pre-training remain
out-of-scope. See `internal-docs/oos/2026/08/oos-202608011322-scope.md`.
