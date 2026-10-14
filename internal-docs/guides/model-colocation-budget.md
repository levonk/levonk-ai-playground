# Model Co-location Budget

> Which models can run simultaneously within a VRAM budget, and how to plan
> co-located deployments.
>
> Sources: [GPUpartner](https://www.gpupartner.com/blog/a-practical-guide-to-running-8x-rtx-pro-6000s) and [lws.io](https://lws.io/blog/my-local-model-setup/)

## The constraint

Each model script binds a unique host port, so multiple models can run
simultaneously. The only constraint is total VRAM:

```
Sum of (model weights + KV cache allocation) across all running models ≤ Available VRAM
```

## GB10 128GB co-location table

Assumes ~6 GB runtime overhead, leaving ~122 GB for models + KV cache.
Weights are approximate for AWQ-4bit / GPTQ-int4 quantization.

| Combination | Weights total | KV cache room | Feasible? | Notes |
|-------------|---------------|---------------|-----------|-------|
| 1× 80B-A3B (chat) | ~45 GB | ~77 GB | ✅ | Default single-model setup |
| 1× 80B-A3B + 1× 4B lite | ~47.5 GB | ~74.5 GB | ✅ | Recommended: big + lite |
| 1× 80B-A3B + 1× 35B-A3B | ~65 GB | ~57 GB | ✅ | Chat + reasoning |
| 1× 80B-A3B + 1× 80B-A3B | ~90 GB | ~32 GB | ⚠️ Tight | Two big models, limited KV cache |
| 1× 80B-A3B + 1× 80B-A3B + 1× 4B | ~92.5 GB | ~29.5 GB | ⚠️ Very tight | Not recommended |
| 3× 80B-A3B | ~135 GB | <0 | ❌ | Exceeds VRAM |
| 1× 80B-A3B + 1× 35B-A3B + 1× 4B + TTS + STT | ~68 GB | ~54 GB | ✅ | Full stack |
| 1× 80B-A3B + TTS + STT | ~46 GB | ~76 GB | ✅ | Chat + audio services |

### TTS and STT are tiny

Kokoro-82M (TTS) uses ~0.3 GB and Whisper large-v3-turbo (STT) uses ~3 GB.
They barely affect the co-location budget and can run alongside any LLM
configuration.

## Apple Silicon 48GB co-location table

Assumes ~8 GB OS overhead, leaving ~40 GB for models + KV cache.

| Combination | Weights total | KV cache room | Feasible? |
|-------------|---------------|---------------|-----------|
| 1× 35B-A3B 4-bit | ~20 GB | ~20 GB | ✅ |
| 1× 35B-A3B + 1× 4B lite | ~22.4 GB | ~17.6 GB | ✅ |
| 1× 70B dense 4-bit | ~40 GB | <0 | ❌ (swaps to SSD) |
| 1× 27B dense 4-bit | ~14 GB | ~26 GB | ✅ |
| 1× 27B dense + 1× 4B lite | ~16.4 GB | ~23.6 GB | ✅ |

## Multi-GPU PCIe (8× 96GB) co-location

Each GPU is independent (TP=1). The budget is per-GPU, not node-wide.

| Per-GPU config | Weights/GPU | KV cache room/GPU | Concurrent @ 4k |
|---------------|-------------|-------------------|-----------------|
| 1× 27B FP8 | ~27 GB | ~59 GB | ~287 |
| 1× 70B FP8 | ~70 GB | ~16 GB | ~25 |
| 3× 8B FP8 | ~24 GB | ~62 GB | ~244 |
| 1× 32B FP8 | ~32 GB | ~54 GB | ~106 |

With 8 GPUs, multiply concurrent sessions by 8. See
[NVIDIA Multi-GPU PCIe](../platforms/nvidia-multi-gpu-pcie.md) for fleet patterns.

## Planning your deployment

1. **Identify your VRAM budget** — see your platform doc.
2. **List the models you want to run** — check weights from the model table.
3. **Sum the weights** — ensure total < available VRAM with 10–15% buffer.
4. **Check KV cache room** — remaining VRAM after weights is for KV cache.
5. **Decide on FP8 KV cache** — if KV cache room is tight, enable FP8 to double it.
6. **Assign ports** — each model gets a unique port (8000–8006 in this repo).

## Example: full-stack GB10 deployment

```bash
# Terminal 1: Chat model (port 8000, ~45 GB)
./scripts/model-chat.sh

# Terminal 2: Code model (port 8001, ~45 GB)
# Note: 45+45 = 90 GB, leaves ~32 GB for KV cache split across both
./scripts/model-code.sh

# Terminal 3: Lite model (port 8004, ~2.5 GB)
# Tiny footprint, runs alongside anything
./scripts/model-lite.sh

# Terminal 4: TTS (port 8005, ~0.3 GB)
./scripts/model-tts.sh

# Terminal 5: STT (port 8006, ~3 GB)
./scripts/model-stt.sh
```

Total: ~95.8 GB weights, ~26 GB for KV cache. Feasible but tight. If you
need more KV cache headroom, drop one of the 80B models or use FP8 KV cache.
