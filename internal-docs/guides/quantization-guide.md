# Quantization Guide

> Trade-offs between quantization formats (AWQ, GPTQ, OptiQ, FP8, BF16) and
> which runtimes support each.
>
> Sources: [lws.io](https://lws.io/blog/my-local-model-setup/) and [GPUpartner](https://www.gpupartner.com/blog/a-practical-guide-to-running-8x-rtx-pro-6000s)

## Quick comparison

| Format | Bits | Quality vs BF16 | VRAM (per 1B params) | Supported runtimes |
|--------|------|-----------------|----------------------|-------------------|
| BF16 | 16 | Baseline (100%) | 2 GB | All |
| FP8 | 8 | ~98–99% | 1 GB | vLLM, SGLang, FreeToken |
| AWQ-4bit | 4 | ~95–97% | 0.5 GB | SGLang, vLLM |
| GPTQ-int4 | 4 | ~94–96% | 0.5 GB | vLLM only |
| OptiQ-4bit | 4 (mixed) | ~96–98% | ~0.55 GB | Check per model |
| GGUF Q4_K_M | 4 | ~94–96% | ~0.55 GB | llama.cpp, vLLM (limited) |

## Format details

### BF16 (Bfloat16)

- **What it is**: 16-bit floating point, the uncompressed baseline
- **Quality**: Best — no quantization artifacts
- **VRAM**: 2 GB per billion parameters (70B model = 140 GB)
- **Use when**: You have abundant VRAM and need maximum quality
- **Runtimes**: All (SGLang, vLLM, FreeToken, MLX)

### FP8 (8-bit floating point)

- **What it is**: 8-bit floating point (e4m3 or e5m2 format)
- **Quality**: ~98–99% of BF16 — nearly lossless
- **VRAM**: 1 GB per billion parameters (70B model = 70 GB)
- **Use when**: You have enough VRAM for 8-bit and want near-BF16 quality
- **Runtimes**: vLLM, SGLang, FreeToken
- **Note**: Blackwell GPUs (GB10, RTX PRO 6000) have native FP8 support,
  making this very fast. On older GPUs, FP8 may be emulated.

### AWQ-4bit (Activation-aware Weight Quantization)

- **What it is**: 4-bit quantization that preserves important weights based
  on activation patterns. Sensitive layers get higher precision.
- **Quality**: ~95–97% of BF16 — small degradation, imperceptible for most tasks
- **VRAM**: 0.5 GB per billion parameters (80B model = ~45 GB with overhead)
- **Use when**: VRAM is limited and you need to fit a large model
- **Runtimes**: SGLang, vLLM
- **Note**: This is the primary quantization used in this repo for GB10.
  The `--quantization awq_merlin` flag is passed to vLLM/SGLang.

### GPTQ-int4

- **What it is**: Post-training quantization using second-order error
  compensation. 4-bit integer weights.
- **Quality**: ~94–96% of BF16 — slightly below AWQ in some benchmarks
- **VRAM**: 0.5 GB per billion parameters
- **Use when**: AWQ variant isn't available for your model
- **Runtimes**: vLLM only (SGLang and FreeToken do not support GPTQ)
- **Note**: This repo uses GPTQ for the reasoning model
  (`Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-int4`) because
  no AWQ variant was available. This forces `RUNTIME=vllm` instead of SGLang.

### OptiQ-4bit (mixed-precision)

- **What it is**: 4-bit quantization with 8-bit precision on sensitive
  layers. Adaptive precision allocation.
- **Quality**: ~96–98% of BF16 — better than uniform 4-bit, close to FP8
- **VRAM**: ~0.55 GB per billion parameters (slightly higher than AWQ due
  to mixed precision)
- **Use when**: You want better quality than AWQ-4bit but can't afford FP8
- **Runtimes**: Check per model — OptiQ is newer and runtime support varies
- **Note**: The lws.io blog reports OptiQ-4bit loses only 1–2 points vs BF16
  on benchmarks. If available for your model and runtime, it's a strong
  choice. Verify compatibility before committing — this repo has an
  anti-pattern entry on Intel AutoRound int4 on GB10, showing that not all
  4-bit quantizations work well on Blackwell.

### GGUF (llama.cpp format)

- **What it is**: Quantized format for llama.cpp. Multiple quantization
  levels (Q2_K through Q8_0).
- **Quality**: Varies by level (Q4_K_M ≈ AWQ-4bit quality)
- **VRAM**: ~0.55 GB per billion parameters (Q4_K_M)
- **Use when**: Running on CPU, Vulkan GPU, or via llama.cpp
- **Runtimes**: llama.cpp, vLLM (limited support), not SGLang/FreeToken
- **Note**: This repo uses GGUF as the fallback for Vulkan systems.

## Choosing a quantization for your model

```
1. Check what quantizations are available for your model on HuggingFace.
2. Check which runtimes support that quantization (see table above).
3. Pick the highest-precision format that fits your VRAM budget:

   BF16  → if VRAM allows (2 GB/B params)
   FP8   → if VRAM allows (1 GB/B params) and GPU supports it
   OptiQ → if available and runtime supports it (~0.55 GB/B params)
   AWQ   → if SGLang or vLLM (0.5 GB/B params)
   GPTQ  → if only vLLM and no AWQ available (0.5 GB/B params)
   GGUF  → if CPU/Vulkan/llama.cpp only
```

## This repo's quantization choices

| Script | Model | Quant | Runtime | Why |
|--------|-------|-------|---------|-----|
| model-chat.sh | Qwen3-Next-80B-A3B-Instruct | AWQ-4bit | SGLang | Best VRAM/quality for 80B on 128 GB |
| model-code.sh | Qwen3-Coder-Next | AWQ-4bit | SGLang | Same reasoning |
| model-reason.sh | Qwen3.5-35B-A3B-Reasoning | GPTQ-int4 | vLLM | No AWQ variant available; GPTQ forces vLLM |
| model-general.sh | Qwen3-Next-80B-A3B-Thinking | AWQ-4bit | vLLM | Speculative decoding needs vLLM |
| model-lite.sh | Qwen3-4B-Instruct | AWQ-4bit | SGLang | Small model, AWQ for consistency |

If an OptiQ-4bit or FP8 variant becomes available for any of these models,
consider switching — FP8 would give better quality at 2× the VRAM (still
fits on 128 GB for the 80B models), and OptiQ would give better quality at
similar VRAM. Always benchmark before committing.
