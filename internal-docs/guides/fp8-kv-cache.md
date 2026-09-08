# FP8 KV Cache

> FP8 KV cache quantization halves the memory needed per token's KV cache,
> doubling concurrent session capacity with minimal quality impact.
>
> Source: [GPUpartner — 8x RTX PRO 6000 guide](https://www.gpupartner.com/blog/a-practical-guide-to-running-8x-rtx-pro-6000s)

## What it does

The KV cache stores the key and value tensors for every token in every
active sequence. By default, these are stored in BF16 (2 bytes per element).
FP8 KV cache stores them in 8-bit floating point (1 byte per element), halving
memory usage.

## Quality impact

FP8 KV cache typically loses 1–2 points on benchmarks compared to BF16 KV
cache. For most workloads (chat, coding, reasoning) this is imperceptible.
For workloads requiring maximum precision (e.g. mathematical computation,
exact retrieval), BF16 may be preferable.

## How to enable

### vLLM

```bash
# In model scripts, pass --kv-cache-dtype fp8 to run_container:
run_container "$ACCT" "$MODEL" "$HOST_PORT" \
  --kv-cache-dtype fp8

# Or via environment variable:
# The runner-lib.sh passes additional_args through to vLLM, so you can
# set it in the model script or override at runtime:
KV_CACHE_DTYPE=fp8 ./model-chat.sh
```

vLLM also supports `fp8_e5m2` and `fp8_e4m3` variants. The default `fp8`
uses `fp8_e4m3` which is the recommended choice for KV cache.

### SGLang

SGLang does not yet have a direct `--kv-cache-dtype fp8` flag in all
versions. Check the SGLang docs for your version. In some versions, FP8 KV
cache is enabled automatically when the model is loaded in FP8 precision.

### MLX (vllm-mlx)

MLX uses its own KV cache management. Check `vllm-mlx` docs for FP8 KV
cache support — it may be enabled via `--kv-cache-dtype` or a similar flag.

## When to use FP8 KV cache

| Scenario | Recommendation |
|----------|---------------|
| High-concurrency chat serving | Enable FP8 — doubles capacity |
| Long-context document analysis | Enable FP8 — fits more tokens |
| Agent workflows with tool use | Enable FP8 — long context chains |
| Mathematical/scientific computation | Use BF16 — precision matters |
| Small models with plenty of VRAM | Optional — you may not need it |

## How this repo uses it

The `runner-lib.sh` does not currently pass `--kv-cache-dtype` by default.
To enable FP8 KV cache for a specific model, add it to the model script's
`run_container` call:

```bash
# In model-chat.sh, add --kv-cache-dtype fp8:
run_container "$ACCT" "$MODEL" "$HOST_PORT" \
  --kv-cache-dtype fp8
```

Or set it globally via an environment variable and have the model script
pass it conditionally. See the updated `runner-lib.sh` for the
`KV_CACHE_DTYPE` environment variable support.

## Capacity comparison

For Qwen3-Next-80B-A3B on GB10 128GB (AWQ-4bit weights, ~45 GB):

| KV precision | Usable VRAM | KV/token | Concurrent @ 4k | Concurrent @ 32k |
|-------------|-------------|----------|-----------------|-------------------|
| BF16 | ~77 GB | 64 KB | ~300 | ~37 |
| FP8 | ~77 GB | 32 KB | ~600 | ~75 |
| INT4 (if supported) | ~77 GB | 16 KB | ~1,200 | ~150 |

FP8 doubles capacity with minimal quality loss. INT4 would quadruple it
but is not universally supported and has more noticeable quality impact.
