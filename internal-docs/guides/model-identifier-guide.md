# Model Identifier Guide

> How to read model names to understand parameter count, MoE structure,
> quantization, and instruction tuning — so you can predict VRAM usage and
> quality before downloading.
>
> Source: [lws.io — My local model setup](https://lws.io/blog/my-local-model-setup/)

## Anatomy of a model identifier

Model names follow a loose convention. Here's how to decode them.

### Example 1: `Qwen3.6-35B-A3B-OptiQ-4bit`

| Part | Meaning |
|------|---------|
| `Qwen3.6` | Model family and version |
| `35B` | Total parameters across all experts (35 billion) |
| `A3B` | Active parameters per token (3 billion, not 35) |
| `OptiQ` | Mixed-precision quantization method |
| `4bit` | Target precision (4-bit mostly, 8-bit on sensitive layers) |

### Example 2: `gemma-4-e4b-it-4bit`

| Part | Meaning |
|------|---------|
| `gemma-4` | Google's Gemma 4 family |
| `e4b` | Encoding size, roughly 4 billion parameters total |
| `it` | Instruction-tuned (fine-tuned for chat/instruction following) |
| `4bit` | Uniform 4-bit quantization |

### Example 3: `Qwen3-Next-80B-A3B-Instruct-AWQ-4bit`

| Part | Meaning |
|------|---------|
| `Qwen3-Next` | Model family (Qwen3, Next generation) |
| `80B` | Total parameters (80 billion) |
| `A3B` | Active parameters per token (3 billion) |
| `Instruct` | Instruction-tuned |
| `AWQ` | Activation-aware Weight Quantization method |
| `4bit` | 4-bit precision |

### Example 4: `Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-int4`

| Part | Meaning |
|------|---------|
| `Qwen3.5-35B-A3B` | Base model (Qwen 3.5, 35B total, 3B active) |
| `Claude-4.6-Opus` | Distilled from Claude 4.6 Opus (teacher model) |
| `Reasoning` | Fine-tuned for reasoning/chain-of-thought |
| `Distilled` | Knowledge distillation from a larger teacher model |
| `GPTQ` | GPTQ quantization method |
| `int4` | 4-bit integer quantization |

## Dense vs MoE — the critical distinction

### Dense models

A dense model activates ALL parameters for every token. A 27B dense model
loads 27 billion parameters in VRAM and computes all 27 billion for every
token.

```
VRAM needed ≈ parameter_count × bytes_per_param
27B dense in 4-bit ≈ 27 × 0.5 = 13.5 GB
27B dense in FP8   ≈ 27 × 1   = 27 GB
27B dense in BF16  ≈ 27 × 2   = 54 GB
```

### MoE (Mixture of Experts) models

An MoE model has many "experts" (sub-networks), but only activates a subset
per token. The `A` in `A3B` means "active" — only 3 billion parameters are
computed per token, even though all 35 billion must be loaded in VRAM.

```
VRAM needed ≈ TOTAL parameter_count × bytes_per_param (all experts loaded)
Compute per token ≈ ACTIVE parameter_count × bytes_per_param
```

**Qwen3.5-35B-A3B in 4-bit:**
- VRAM: 35 × 0.5 ≈ 17.5 GB (all 35B loaded)
- Compute per token: 3B parameters (fast, like a 6B dense model)

**Qwen3-Next-80B-A3B in 4-bit:**
- VRAM: 80 × 0.5 ≈ 40 GB (all 80B loaded)
- Compute per token: 3B parameters (same speed as the 35B model!)

This is why MoE models are so attractive: you get the quality of a large
model (more total parameters = more knowledge) with the speed of a small
model (fewer active parameters = faster inference). The trade-off is VRAM:
you must load ALL experts even though only a few are active per token.

## Why this matters for hardware selection

| Model | Type | VRAM (4-bit) | Speed like a... | Fits on... |
|-------|------|-------------|-----------------|------------|
| Qwen3-4B | Dense | ~2 GB | 4B dense | Anything ≥8 GB |
| Qwen3.5-35B-A3B | MoE | ~20 GB | 6B dense | 32 GB+ |
| Qwen3-Next-80B-A3B | MoE | ~45 GB | 6B dense | 64 GB+ |
| Llama-3.3-70B | Dense | ~40 GB | 70B dense | 64 GB+ |
| Llama-3.1-405B | Dense | ~200 GB | 405B dense | 512 GB+ |

The 80B-A3B MoE and the 70B dense both need ~40–45 GB of VRAM, but the MoE
generates tokens at the speed of a 6B model while the dense 70B generates at
the speed of a 70B model. For chat and coding, the MoE is almost always the
better choice when VRAM allows.

## Quantization suffixes

| Suffix | Method | Quality | VRAM | Runtime support |
|--------|--------|---------|------|-----------------|
| `AWQ-4bit` | Activation-aware Weight Quantization | Good | Low | SGLang, vLLM |
| `GPTQ-int4` | GPTQ (post-training quantization) | Good | Low | vLLM only |
| `OptiQ-4bit` | Mixed-precision (4-bit + 8-bit sensitive layers) | Better | Slightly higher | Check per runtime |
| `FP8` | 8-bit floating point | Excellent | Medium | vLLM, SGLang, FreeToken |
| `BF16` | 16-bit (uncompressed baseline) | Best | High | All runtimes |
| `4bit` (generic) | Usually AWQ or GPTQ | Good | Low | Runtime-dependent |
| `GGUF` | llama.cpp format, various quant levels | Good | Low | vLLM (limited), llama.cpp |

See [Quantization Guide](quantization-guide.md) for detailed trade-offs.

## Other common suffixes

| Suffix | Meaning |
|--------|---------|
| `Instruct` | Instruction-tuned for chat/following instructions |
| `it` | Instruction-tuned (same as above, different naming) |
| `Chat` | Chat-tuned variant |
| `Coder` | Fine-tuned for code generation |
| `Reasoning` | Fine-tuned for chain-of-thought reasoning |
| `Thinking` | Similar to Reasoning; may include extended thinking tokens |
| `Distilled` | Knowledge distilled from a larger teacher model |
| `Turbo` | Optimized for speed (fewer layers, distillation, or pruning) |
| `mlx` or `MLX` | Pre-converted to Apple MLX format |
| `GGUF` | llama.cpp format (CPU/GPU via llama.cpp) |
