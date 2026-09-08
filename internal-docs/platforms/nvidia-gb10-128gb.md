# NVIDIA GB10 128GB (DGX Spark)

> The primary target platform for this repo. Single Blackwell GPU with 128 GB
> GDDR7. All model scripts default to this hardware.

## Specs

| Field | Value |
|-------|-------|
| GPU | NVIDIA GB10 (Blackwell) |
| VRAM | 128 GB GDDR7 |
| Interconnect | N/A (single GPU) |
| Tensor parallelism | TP=1 always (single GPU) |
| Default runtime | SGLang (dense), FreeToken (MoE) |
| Fallback runtime | vLLM (GPTQ, compressed-tensors, multi-GPU PP) |
| Docker image | `lmsysorg/sglang:v0.5.16-cu130`, `nvcr.io/nvidia/vllm:26.04-py3` |

## VRAM budget

| Component | VRAM | Notes |
|-----------|------|-------|
| CUDA context + runtime overhead | ~4 GB | Allocated at startup, not available for models |
| SGLang/vLLM framework overhead | ~2 GB | Paged allocator, scheduler, etc. |
| **Available for models + KV cache** | **~122 GB** | At default `GPU_MEMORY_UTILIZATION=0.90` |
| At `GPU_MEMORY_UTILIZATION=0.95` | ~129 GB | Tighter margin; use only if model + KV cache fits comfortably |

`GPU_MEMORY_UTILIZATION` controls the fraction of VRAM the runtime pre-allocates.
At 0.90, the runtime reserves 90% of 128 GB (~115 GB) for model weights + KV cache,
leaving the rest for CUDA context and framework overhead. At 0.95 you get more
room for models but risk OOM if KV cache grows unexpectedly.

## Model sizing

| Model | Weights (4-bit) | Weights (FP8) | Weights (BF16) | Fits 128GB? |
|-------|----------------|---------------|----------------|-------------|
| Qwen3-4B (lite) | ~2.5 GB | ~4 GB | ~8 GB | Yes, trivially |
| Qwen3.5-35B-A3B (MoE) | ~20 GB | ~35 GB | ~70 GB | Yes (4-bit/FP8) |
| Qwen3-Next-80B-A3B (MoE) | ~45 GB | ~80 GB | ~160 GB | Yes (4-bit only) |
| Llama-3.3-70B (dense) | ~40 GB | ~70 GB | ~140 GB | Yes (4-bit/FP8) |
| Llama-3.1-405B (dense) | ~200 GB | ~405 GB | ~810 GB | No |

For MoE models, the **total parameter count** determines VRAM needed (all
experts must be loaded), but only the **active parameters** determine compute
speed. See [Model Identifier Guide](../guides/model-identifier-guide.md).

## Co-location

Multiple models can run simultaneously since each script binds a unique host
port. The constraint is total VRAM across all running models.

| Combination | Total weights | KV cache room | Feasible? |
|-------------|--------------|---------------|-----------|
| 1× 80B-A3B AWQ-4bit | ~45 GB | ~77 GB | Yes |
| 1× 80B-A3B + 1× 4B lite | ~47.5 GB | ~74.5 GB | Yes |
| 2× 80B-A3B AWQ-4bit | ~90 GB | ~32 GB | Tight, possible |
| 1× 80B-A3B + 1× 35B-A3B | ~65 GB | ~57 GB | Yes |
| 3× 80B-A3B AWQ-4bit | ~135 GB | <0 | No |

See [Model Co-location Budget](../guides/model-colocation-budget.md) for the
general formula and more combinations.

## Recommended runtimes

| Model type | Runtime | Why |
|-----------|---------|-----|
| Dense, in-VRAM, AWQ/FP8 | SGLang | Best throughput on Blackwell for dense models |
| MoE, in-VRAM | FreeToken | MoE-optimized, native Python, no Docker overhead |
| GPTQ quantization | vLLM | SGLang/FreeToken lack GPTQ support |
| Compressed-tensors | vLLM | vLLM has mature compressed-tensors support |
| Multi-GPU pipeline parallel | vLLM | Only vLLM supports PP on this repo's runtimes |

See [ADR-202608021744](../adr/2026/08/adr-202608021744-sglang-glom-runtime-mapping.md)
for the full runtime selection rationale.

## KV cache

At 128 GB with a 45 GB model (80B-A3B AWQ-4bit), ~77 GB is available for KV
cache. With FP8 KV cache quantization (32 KB/token for Qwen3 architecture),
that supports ~2.4 million tokens of concurrent context. With BF16 KV cache
(64 KB/token), ~1.2 million tokens.

See [KV Cache Capacity Planning](../guides/kv-cache-capacity-planning.md) and
[FP8 KV Cache](../guides/fp8-kv-cache.md) for details.

## Port assignments

| Script | Port | Model |
|--------|------|-------|
| `model-chat.sh` | 8000 | Qwen3-Next-80B-A3B-Instruct-AWQ-4bit |
| `model-code.sh` | 8001 | Qwen3-Coder-Next-AWQ-4bit |
| `model-reason.sh` | 8002 | Qwen3.5-35B-A3B-Reasoning-Distilled-GPTQ-int4 |
| `model-general.sh` | 8003 | Qwen3-Next-80B-A3B-Thinking-AWQ-4bit |
| `model-lite.sh` | 8004 | Qwen3-4B-Instruct (lightweight companion) |
| `model-tts.sh` | 8005 | Kokoro-82M (text-to-speech) |
| `model-stt.sh` | 8006 | Whisper large-v3-turbo (speech-to-text) |

## SSD considerations

The GB10 DGX Spark typically has NVMe storage. SSD wear is less of a concern
than on Apple Silicon (where the SSD is soldered and non-replaceable), but
model downloads still write 20–80 GB per model. If you swap models frequently,
consider pointing `LOCAL_CACHE_DIR` to an external NVMe SSD to reduce writes
on the internal drive.

See [SSD Wear Mitigation](../guides/ssd-wear-mitigation.md) for details.
