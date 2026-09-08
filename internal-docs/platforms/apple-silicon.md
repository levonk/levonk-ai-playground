# Apple Silicon Mac (M1–M5)

> Apple Silicon Macs use unified memory — the GPU and CPU share the same RAM
> pool. This eliminates the PCIe bottleneck but introduces SSD wear concerns
> since the SSD is soldered and non-replaceable.

## Specs by chip tier

| Chip | Unified RAM | Memory bandwidth | Typical use |
|------|------------|-------------------|-------------|
| M1/M2/M3 base | 8–16 GB | 100–200 GB/s | Small models only (≤7B 4-bit) |
| M1/M2/M3 Pro | 16–36 GB | 200–270 GB/s | Mid-size models (≤14B 4-bit) |
| M4 Pro | 48 GB | 273 GB/s | 35B-A3B MoE 4-bit (20 GB), leaves 28 GB |
| M2/M3/M4 Max | 64–128 GB | 400–546 GB/s | 70B dense 4-bit or multiple MoE |
| M2/M3 Ultra | 128–192 GB | 800 GB/s | 70B+ dense, multiple large MoE |

## Runtime

Apple Silicon uses the **MLX** runtime via `vllm-mlx`:

```bash
pip install vllm-mlx
```

MLX provides:
- Continuous batching
- Paged KV cache
- Prefix caching
- SSD-tiered KV cache (blocks cached to SSD between sessions)
- OpenAI + Anthropic API compatibility
- Multimodal model support

Models from `mlx-community` on HuggingFace are pre-converted to MLX format.
Other safetensors checkpoints are auto-converted on first load (this writes
new files — see [SSD Wear Mitigation](../guides/ssd-wear-mitigation.md)).

## VRAM budget

Unlike discrete GPUs, Apple Silicon shares RAM with the OS. The budget is:

```
Available for models = Total unified RAM - OS overhead - app overhead
```

| RAM | OS overhead | Available for models | Notes |
|-----|-----------|---------------------|-------|
| 16 GB | ~6–8 GB | ~8–10 GB | Only small models (≤7B 4-bit) |
| 32 GB | ~8 GB | ~24 GB | Mid-size (≤14B 4-bit, 35B-A3B MoE 4-bit tight) |
| 48 GB | ~8 GB | ~40 GB | 35B-A3B MoE 4-bit (20 GB) + 20 GB KV cache |
| 64 GB | ~8 GB | ~56 GB | 70B dense 4-bit (40 GB) + 16 GB KV cache |
| 128 GB | ~8 GB | ~120 GB | Multiple large models |
| 192 GB | ~8 GB | ~184 GB | 70B+ dense, multiple MoE |

### How to check if a model will fit

1. Look at the quantized file size. A 4-bit model is roughly the parameter
   count in gigabytes (35B params ≈ 17–20 GB depending on quantization method).
2. Subtract OS overhead (6–8 GB on macOS).
3. Leave room for KV cache. Every few thousand tokens adds megabytes to the
   KV cache. Plan for 8–16 GB overhead for long conversations.
4. For MoE models, the total parameter count is misleading. Look for the
   "active parameters" figure to understand actual inference memory.
5. If your model plus context fits within available unified memory with a
   10–15% buffer, you are good. Anything closer to full will swap to SSD.

### MoE advantage on constrained Macs

A dense 27B model in 4-bit needs ~14 GB — nearly all of a 16 GB Mac's RAM.
An MoE model like Qwen3.5-35B-A3B has 35B total parameters but only 3B active
per token. In 4-bit it takes ~20 GB of unified memory (all experts loaded),
but the inference memory footprint is closer to a 6B dense model. This means
it can run on a 32 GB Mac where a dense 35B model would not.

## SSD wear — critical concern

Apple Silicon Macs have soldered, non-replaceable SSDs. Write endurance is
finite (~3,000 erase cycles for recent models). The main write sources are:

1. **Model downloads** — 2–80 GB per model, write-once
2. **MLX auto-conversion** — converting safetensors to MLX format writes new files
3. **KV cache SSD tiering** — vllm-mlx/oMLX cache KV blocks to SSD (high-write, repeated)
4. **Docker VM image** — Docker Desktop's VM disk (if using Docker)
5. **Swap** — macOS compresses RAM pages before swapping; chronic memory pressure causes swap writes

**Recommendation:** Move model storage to an external NVMe SSD (USB-C /
Thunderbolt 4/5). Model weights are write-once-read-many — perfect for
external storage. See [SSD Wear Mitigation](../guides/ssd-wear-mitigation.md)
for the full strategy.

## oMLX vs vllm-mlx

The repo defaults to `vllm-mlx` for Apple Silicon. An alternative is
[oMLX](https://omlx.ai), which offers:

- Built-in HuggingFace model browser (download models from a web UI)
- SSD-tiered KV cache persistence (caches blocks to disk, restores in
  milliseconds when agents revisit earlier context)
- Auto-discovery of models in `~/models/`
- Admin dashboard for model management

oMLX's KV cache persistence is particularly valuable for agent workflows
(coding agents, Hermes) that repeatedly circle back through earlier context.
When the agent returns to a previous prefix, the KV cache is restored from
disk in milliseconds instead of being recomputed.

If you primarily run agent workloads on Apple Silicon, consider oMLX as an
alternative to vllm-mlx. The repo's `RUNTIME=mlx` path uses vllm-mlx; to use
oMLX, run it directly and point your clients at its port.

## Performance expectations

| Chip | Model | Prompt processing | Token generation |
|------|-------|-------------------|-----------------|
| M4 Pro (48 GB) | Qwen3.5-35B-A3B 4-bit | ~325 tok/s | ~34 tok/s |
| M4 Pro (48 GB) | Qwen3-4B 4-bit | ~500+ tok/s | ~80+ tok/s |
| M3 Max (128 GB) | Qwen3-Next-80B-A3B 4-bit | ~200 tok/s | ~25 tok/s |

Memory bandwidth is the speed limit. The M4 Pro has 273 GB/s; M3 Max has
400 GB/s; M2 Ultra has 800 GB/s. Token generation speed scales roughly
linearly with bandwidth.

## Networking

For multi-device access (phone, laptop, desktop hitting the same model
server), use [Tailscale](https://tailscale.com) to create a private mesh
network. No ports exposed to the public internet. See
[Networking Guide](../guides/networking-guide.md).

## Port assignments

Same as GB10 — each model script binds a unique port. On Apple Silicon the
scripts use `RUNTIME=mlx` instead of Docker-based runtimes.

| Script | Port | Model (Apple Silicon) |
|--------|------|-----------------------|
| `model-chat.sh` | 8000 | mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit |
| `model-code.sh` | 8001 | mlx-community/Qwen3-Coder-Next-4bit |
| `model-reason.sh` | 8002 | mlx-community/Qwen3.5-35B-A3B-Instruct-4bit |
| `model-general.sh` | 8003 | mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit |
| `model-lite.sh` | 8004 | mlx-community/Qwen3-4B-Instruct-4bit |
| `model-tts.sh` | 8005 | Kokoro-82M (CPU/MPS) |
| `model-stt.sh` | 8006 | Whisper large-v3 (CPU/MPS) |
