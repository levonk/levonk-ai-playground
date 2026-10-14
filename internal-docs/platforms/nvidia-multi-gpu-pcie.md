# NVIDIA Multi-GPU PCIe (e.g. 8x RTX PRO 6000)

> Multi-GPU systems connected via PCIe Gen 5 (no NVLink). The key insight from
> the [GPUpartner guide](https://www.gpupartner.com/blog/a-practical-guide-to-running-8x-rtx-pro-6000s):
> **prefer isolated TP=1 serving over model sharding**. PCIe bandwidth is too
> low for efficient tensor parallelism on large models, but perfect for running
> many independent models in parallel.

## Specs (example: 8x RTX PRO 6000 Blackwell + EPYC 9555)

| Field | Value |
|-------|-------|
| GPUs | 8 × NVIDIA RTX PRO 6000 Blackwell |
| VRAM per GPU | 96 GB GDDR7 |
| Total VRAM | 768 GB |
| Interconnect | PCIe Gen 5 x16 (64 GB/s unidirectional, 128 GB/s bidirectional) |
| Host CPU | AMD EPYC 9555 (64-core Zen 5, 128 threads) |
| Host memory | 12-channel DDR5 (~576–614 GB/s) |

## Interconnect comparison

| Interconnect | Unidirectional | Bidirectional | AllReduce latency | Relative cost |
|-------------|----------------|---------------|-------------------|---------------|
| PCIe Gen 5 x16 | 64 GB/s | 128 GB/s | 12–25 μs | $$ |
| NVLink 4 (Hopper) | 450 GB/s | 900 GB/s | 1.5–3.0 μs | $$$ |
| NVLink 5 (Blackwell) | 900 GB/s | 1,800 GB/s | 1.0–2.0 μs | $$$$ |

PCIe Gen 5 is ~7× slower than NVLink 4 and ~14× slower than NVLink 5. This
matters enormously for tensor parallelism (TP>1), where AllReduce operations
happen on every forward pass. For TP=1 (isolated serving), interconnect
bandwidth is irrelevant — each GPU runs independently.

## Recommended strategy: TP=1, DP=N

Run independent single-GPU instances. Each GPU hosts one model with no
inter-card communication.

### Why TP=1 wins on PCIe

- **Zero inter-GPU bus traffic** — no AllReduce, no synchronization stalls
- **Minimal inter-token latency** — no waiting for cross-card gradients
- **Isolated memory pools** — each GPU's VRAM is fully dedicated to its model
- **Linear throughput scaling** — 8 GPUs = 8× throughput, no diminishing returns

### When to shard (TP>1) anyway

Only when a single model is too large for one GPU and you have no NVLink
alternative. Accept the latency penalty:

- Llama-3.1-405B in FP8 (~200 GB) needs 3 GPUs at TP=3
- DeepSeek-R1-671B in 4-bit (~350 GB) needs 4+ GPUs at TP=4

For these sizes, consider whether an NVLink system (HGX H200/B200) is more
appropriate. See [NVIDIA HGX / NVLink](nvidia-hgx-nvlink.md).

## KV cache density (per 96 GB GPU)

| Model | Weights | KV precision | Usable VRAM | KV/token | Concurrent (4k ctx) |
|-------|---------|-------------|-------------|----------|-------------------|
| Qwen3.8-27B (dense) | FP8 (~27 GB) | FP8 | ~59.4 GB | 32 KB | ~287/card |
| Qwen3.8-27B (dense) | BF16 (~54 GB) | FP8 | ~32.4 GB | 32 KB | ~156/card |
| Qwen3.8-27B (dense) | BF16 (~54 GB) | BF16 | ~32.4 GB | 64 KB | ~97/card |
| Llama-3.3-70B (dense) | FP8 (~70 GB) | FP8 | ~16.4 GB | 160 KB | ~25/card |

With 8 cards: multiply concurrent sessions by 8 (e.g. 2,296 concurrent 4k
sessions for Qwen3.8-27B FP8 across the node).

See [KV Cache Capacity Planning](../guides/kv-cache-capacity-planning.md) for
the formulas behind these numbers.

## Multi-model fleet patterns

| Fleet config | Per-GPU allocation | KV pool | Concurrent (4k) | Endpoints | Use case |
|-------------|-------------------|---------|-----------------|-----------|----------|
| Dedicated 8B | 1× 8B FP8 (~8 GB) | ~78.4 GB | 306/card | 8 models | High-volume single-task API |
| 3× 8B stacking | 3× 8B FP8 (~24 GB) | ~62.4 GB | 244/card | 24 endpoints | Multi-tenant microservice mesh |
| Mid-tier 32B | 1× 32B FP8 (~32 GB) | ~54.4 GB | 106/card | 8 models | Coding & reasoning |

### Mapping to this repo's scripts

On an 8-GPU PCIe system, you can run 8 instances of the repo's model scripts
simultaneously, each pinned to a different GPU via `CUDA_VISIBLE_DEVICES`:

```bash
# GPU 0: chat model
CUDA_VISIBLE_DEVICES=0 ./scripts/model-chat.sh

# GPU 1: code model
CUDA_VISIBLE_DEVICES=1 ./scripts/model-code.sh

# GPU 2: reasoning model
CUDA_VISIBLE_DEVICES=2 ./scripts/model-reason.sh

# GPU 3: general/thinking model
CUDA_VISIBLE_DEVICES=3 ./scripts/model-general.sh

# GPU 4: lite model
CUDA_VISIBLE_DEVICES=4 ./scripts/model-lite.sh

# GPU 5: TTS
CUDA_VISIBLE_DEVICES=5 ./scripts/model-tts.sh

# GPU 6: STT
CUDA_VISIBLE_DEVICES=6 ./scripts/model-stt.sh

# GPU 7: spare / additional model
```

Each script uses `TENSOR_PARALLEL_SIZE=1` (the default). No changes to the
scripts are needed — just set `CUDA_VISIBLE_DEVICES` per instance.

## Fine-tuning (out of scope for this repo, but documented for reference)

This repo is inference-only. However, the 8x 96GB platform supports on-premise
fine-tuning:

| Method | VRAM/card | Host RAM role | Sync bottleneck | Unlocks |
|--------|-----------|--------------|-----------------|---------|
| Full 70B fine-tuning | ~35 GB | ≥1.15 TB (AdamW offload) | PCIe Gen 5 | Full param updates on 70B+ |
| QLoRA | ~35 GB | Dataset staging | Minimal PCIe | Rapid domain adaptation |
| On-node RLHF/DPO | Co-located | Local coordination | Zero network stalls | Actor + critic on one machine |

Full 70B AdamW fine-tuning requires ~840 GB optimizer states offloaded to
host RAM (12× 96 GB DIMMs). At 8 GPUs × 64 GB/s, aggregate PCIe demand is
512 GB/s, which is 83–89% of the EPYC 9555's DDR5 bus — host memory bandwidth
becomes co-limiting.

## Host CPU considerations

The EPYC 9555's 64 Zen 5 cores handle:
- Tokenization across 8–32 parallel inference streams
- JSON grammar enforcement (structured output)
- Embeddings generation
- Network routing
- Prefix-cache staging (host RAM, not active KV paging)

Host RAM should be used for staging, not active KV paging. Paging dynamic
KV caches across the 64 GB/s PCIe boundary during live token generation
causes severe latency spikes.

## Runtime selection

| Model type | Runtime | Why |
|-----------|---------|-----|
| Dense, in-VRAM, TP=1 | SGLang or vLLM | Both work; SGLang preferred for Blackwell |
| MoE, in-VRAM, TP=1 | FreeToken | MoE-optimized |
| GPTQ | vLLM | Only vLLM supports GPTQ |
| Sharded (TP>1) | vLLM | vLLM has the most mature TP support over PCIe |

For TP>1 on this repo's scripts:

```bash
TENSOR_PARALLEL_SIZE=2 ./scripts/model-chat.sh
```

But remember: TP>1 on PCIe introduces AllReduce latency. Only use it when the
model doesn't fit on a single GPU.

See [Tensor Parallelism Guide](../guides/tensor-parallelism-guide.md) for the
TP=1 vs TP>1 decision framework.
