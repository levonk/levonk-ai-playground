# Tensor Parallelism Guide

> When to use TP=1 (isolated serving) vs TP>1 (model sharding), and why the
> interconnect determines the answer.
>
> Source: [GPUpartner — 8x RTX PRO 6000 guide](https://www.gpupartner.com/blog/a-practical-guide-to-running-8x-rtx-pro-6000s)

## What is tensor parallelism?

Tensor parallelism (TP) splits a model's weight matrices across multiple
GPUs. Each GPU computes a portion of each layer's output, then an AllReduce
operation combines the results. This happens on every layer of every
forward pass.

- **TP=1**: The entire model runs on one GPU. No inter-GPU communication.
- **TP=2**: The model is split across 2 GPUs. AllReduce on every layer.
- **TP=8**: The model is split across 8 GPUs. AllReduce on every layer, across all 8.

## The interconnect determines feasibility

| Interconnect | AllReduce latency | TP>1 viable? |
|-------------|-------------------|---------------|
| PCIe Gen 5 x16 | 12–25 μs | Only when model doesn't fit on one GPU |
| NVLink 4 | 1.5–3.0 μs | Yes, efficient |
| NVLink 5 | 1.0–2.0 μs | Yes, very efficient |

On PCIe, each AllReduce takes 12–25 μs. A 70B model has ~80 transformer layers,
so a single forward pass adds ~1–2 ms of pure communication latency. At 30
tokens/second generation, that's 30–60 ms of overhead per second — significant.

On NVLink, the same AllReduce takes 1–3 μs, reducing overhead to ~0.1–0.2 ms
per forward pass — negligible.

## Decision framework

```
Does the model fit on a single GPU (with KV cache room)?
├── YES → Use TP=1. Always. No exceptions.
│         Zero inter-GPU traffic, minimal latency, full VRAM for KV cache.
└── NO → Do you have NVLink?
    ├── YES → Use TP=N (N = ceil(model_size / GPU_VRAM)).
    │        NVLink makes sharding efficient.
    └── NO (PCIe only) → Use TP=N, but accept the latency penalty.
                 Consider whether a smaller model or more aggressive
                 quantization would let you stay at TP=1.
```

## Why TP=1 is better when possible

1. **Zero inter-GPU communication** — no AllReduce, no synchronization stalls
2. **Minimal inter-token latency** — no waiting for cross-card data
3. **Isolated memory pools** — each GPU's VRAM is fully dedicated
4. **Linear throughput scaling** — N GPUs = N× throughput, no diminishing returns
5. **Simpler operations** — no NCCL debugging, no rank failures, no rebalancing

## TP=1 on this repo

All model scripts default to `TENSOR_PARALLEL_SIZE=1`. This is correct for:
- Single-GPU systems (GB10, individual Mac)
- Multi-GPU PCIe systems running independent models per GPU
- Multi-GPU NVLink systems running independent models per GPU

## When to override to TP>1

Only when a single model doesn't fit on one GPU:

```bash
# 405B model on 4 GPUs (NVLink recommended)
TENSOR_PARALLEL_SIZE=4 ./model-chat.sh

# 70B model on 2 GPUs (PCIe, but model needs >96 GB for weights+KV)
TENSOR_PARALLEL_SIZE=2 ./model-chat.sh
```

On NVLink systems, TP>1 is efficient. On PCIe systems, TP>1 works but adds
latency — only use it when there's no alternative.

## Data parallelism vs tensor parallelism

| Strategy | What it does | When to use |
|----------|-------------|-------------|
| TP=1, DP=N | N independent models, one per GPU | High concurrency, multiple models |
| TP=N | One model sharded across N GPUs | Single model too large for one GPU |
| TP=2, DP=N/2 | Pairs of GPUs, each pair runs one model | Medium models that need 2 GPUs each |

For this repo's use case (serving multiple specialized models), TP=1 + DP=N
is almost always the right choice. You get 8 independent endpoints, each with
full VRAM for KV cache.
