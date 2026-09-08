# KV Cache Capacity Planning

> How to calculate how many concurrent sessions your GPU can support based on
> model size, KV cache precision, and available VRAM.
>
> Source: [GPUpartner — 8x RTX PRO 6000 guide](https://www.gpupartner.com/blog/a-practical-guide-to-running-8x-rtx-pro-6000s)

## The formula

```
Usable VRAM = Total VRAM - Model weights - Runtime overhead
KV per token = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
Concurrent sessions = Usable VRAM / (KV per token × context_length)
```

Where `bytes_per_element` is:
- 2 for BF16/FP16
- 1 for FP8/INT8
- 0.5 for FP4/INT4 (if supported)

## Worked examples

### Qwen3.8-27B (dense) on 96 GB GPU, FP8 weights + FP8 KV

| Component | Value |
|-----------|-------|
| Total VRAM | 96 GB |
| Model weights (FP8) | ~27 GB |
| Runtime overhead | ~4 GB |
| Usable VRAM | ~65 GB (conservatively ~59.4 GB) |
| KV/token (FP8) | 32 KB |
| Concurrent @ 4k context | ~287/card |
| Concurrent @ 128k context | ~9/card |
| Concurrent @ 256k context | ~4/card |

### Qwen3.8-27B (dense) on 96 GB GPU, BF16 weights + BF16 KV

| Component | Value |
|-----------|-------|
| Total VRAM | 96 GB |
| Model weights (BF16) | ~54 GB |
| Runtime overhead | ~4 GB |
| Usable VRAM | ~38 GB (conservatively ~32.4 GB) |
| KV/token (BF16) | 64 KB |
| Concurrent @ 4k context | ~97/card |
| Concurrent @ 128k context | ~3/card |

### Qwen3-Next-80B-A3B (MoE) on GB10 128GB, AWQ-4bit weights + FP8 KV

| Component | Value |
|-----------|-------|
| Total VRAM | 128 GB |
| Model weights (AWQ-4bit) | ~45 GB |
| Runtime overhead | ~6 GB |
| Usable VRAM | ~77 GB |
| KV/token (FP8) | ~32 KB (estimated, Qwen3 architecture) |
| Concurrent @ 4k context | ~600 |
| Concurrent @ 32k context | ~75 |
| Concurrent @ 128k context | ~19 |

### Qwen3-4B (lite) on GB10 128GB, AWQ-4bit weights + FP8 KV

| Component | Value |
|-----------|-------|
| Total VRAM | 128 GB |
| Model weights (AWQ-4bit) | ~2.5 GB |
| Runtime overhead | ~6 GB |
| Usable VRAM | ~119.5 GB |
| KV/token (FP8) | ~8 KB (small model, fewer layers) |
| Concurrent @ 4k context | ~3,700+ |
| Concurrent @ 32k context | ~460+ |

The lite model can serve an enormous number of concurrent sessions. This is
why it's ideal for routine tasks — formatting, simple Q&A, throwaway chats.

## FP8 KV cache doubles capacity

Switching from BF16 to FP8 KV cache halves the bytes per element (2 → 1),
which doubles the number of concurrent sessions for the same VRAM budget.

| KV precision | Bytes/element | Relative capacity | Quality impact |
|-------------|---------------|-------------------|----------------|
| BF16 | 2 | 1× | Baseline |
| FP8 | 1 | 2× | Minimal (1–2 points on benchmarks) |
| INT4 | 0.5 | 4× | More noticeable; not universally supported |

See [FP8 KV Cache](fp8-kv-cache.md) for how to enable it.

## Long context vs high concurrency trade-off

There is a direct trade-off: doubling the context length halves the number
of concurrent sessions you can support (assuming fixed VRAM).

| Context length | Concurrent sessions (Qwen3.8-27B FP8, 96 GB) |
|---------------|----------------------------------------------|
| 4k | 287 |
| 32k | 36 |
| 128k | 9 |
| 256k | 4 |

Choose your max context length based on your workload:
- **Chat / Q&A**: 4k–8k is sufficient for most conversations
- **Code generation**: 32k–64k for large file context
- **Document analysis**: 128k+ for full document understanding
- **Agent workflows**: 128k+ for long reasoning chains with tool use

## How to check actual KV cache usage

Both vLLM and SGLang expose metrics endpoints:

```bash
# vLLM — check KV cache utilization
curl http://localhost:8000/metrics | grep vllm:num_preemptions

# SGLang — check KV cache usage
curl http://localhost:8000/metrics | grep kv_cache
```

If preemptions are high, you're running out of KV cache — either reduce
context length, enable FP8 KV cache, or reduce concurrent sessions.
