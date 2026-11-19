# Runtime Selection: Scenario → Runner Matrix

> Which inference engine for which scenario, across every platform this repo
> targets. Extends the tiered model in
> [ADR-202608021744](../adr/2026/08/adr-202608021744-sglang-glom-runtime-mapping.md)
> and its [FreeToken amendment](../adr/2026/08/adr-20260826-freetoken-runtime-amendment.md)
> with the GGUF-native engines (Strata, Basalt). Evaluated 2026-11-18.

## The runners

| Runner | Executes as | API | Strongest at |
|--------|------------|-----|--------------|
| **SGLang** (`lmsysorg/sglang:v0.5.16-cu130`) | Docker | OpenAI | In-VRAM dense models on NVIDIA; AWQ `awq_merlin` kernels |
| **vLLM** (`nvcr.io/nvidia/vllm:26.04-py3`) | Docker | OpenAI | GPTQ/compressed-tensors quants, multi-GPU PP, broadest model support |
| **FreeToken** (`ft`) | native Python | OpenAI + Anthropic | MoE models — semantic KV caching for agent loops, bandwidth-adaptive CPU/GPU expert split |
| **Strata** (`strata`) | native CLI | OpenAI + Anthropic | GGUF-pack MoE on consumer/unified-memory GPUs; **only engine with a tuned linux-aarch64 DGX Spark build** |
| **Basalt** (`basalt`) | native CLI / Docker | OpenAI + Anthropic | Blackwell-tuned Strata fork — **x86-64 + sm_120 only; cannot run on DGX Spark** |
| **MLX** (`vllm-mlx`) | native Python | OpenAI + Anthropic | Apple Silicon via Metal; unified memory, no conversion for mlx-community repos |

## Scenario → runner

| Scenario | Runner | Runner-lib | Why |
|----------|--------|-----------|-----|
| GB10: in-VRAM dense model (AWQ-4bit) | **SGLang** | `RUNTIME=sglang` | `awq_merlin` kernels are the fastest dense-AWQ path on Blackwell |
| GB10: in-VRAM MoE, agentic workload | **FreeToken** | `RUNTIME=freetoken` | Semantic anchor checkpoints reuse KV across agent edits; `fused` backend when experts fit |
| GB10: Qwen3.8-Flash-Next-class GGUF MoE | **Strata** | `RUNTIME=strata` | sm_121 perf table, `--mmap-experts` maps all experts into the unified pool once. Upstream-measured ~43-46 tok/s decode vs ~19-24 estimated for generic runtimes |
| GB10: GPTQ / compressed-tensors / multi-GPU PP | **vLLM** | `RUNTIME=vllm` | FreeToken has no GPTQ support; vLLM is the mature fallback for these formats |
| GB10: frontier MoE exceeding VRAM (753B GLM-5.2 etc.) | **FreeToken** | `RUNTIME=freetoken` | `hybrid`/`offload` backends split experts across GPU + host RAM by calibrated bandwidth |
| x86 Blackwell desktop (RTX 50xx, sm_120) | **Basalt** — or Strata | n/a | Basalt's sm_120-tuned engine + `.basalt` packs; Strata also runs there. **Not reachable via runner-lib yet** — the scripts in this repo are GB10-tuned only |
| Apple Silicon Mac | **MLX** | `RUNTIME=mlx` | Metal-native, pre-converted mlx-community repos; Strata also has a macOS engine but MLX is the repo default |
| AMD/Intel GPU (Vulkan) | **vLLM** (Vulkan backend) or llama.cpp | `RUNTIME=vllm` | No CUDA; GGUF via vLLM's Vulkan path or llama.cpp directly |
| CPU-only | **vLLM** / llama.cpp | `RUNTIME=vllm` | Slow but functional |

## Basalt: evaluated, not adopted (for GB10)

Basalt ([jesdga95/basalt](https://github.com/jesdga95/basalt)) is a Strata fork
optimized for NVIDIA Blackwell — the obvious candidate for the GB10 on paper.
In practice it **cannot run on the DGX Spark**:

- Builds for **Linux/Windows x86-64 with AVX2 only** — the Spark's Grace CPU is
  aarch64. No ARM build exists.
- Targets **sm_120** exclusively — the GB10 GPU is **sm_121**.
- Its killer feature (spilling MoE experts from VRAM into x86 system RAM) is
  also redundant on the Spark, where CPU and GPU already share one 128 GB pool —
  Strata's `--mmap-experts` does the same job without a copy.

**When to revisit:** if Basalt ships a linux-aarch64 build or you add an x86
Blackwell host (RTX 50-series desktop, B200 server) to the fleet.

## Strata on the GB10

Upstream `docs/DGX_SPARK_LINUX.md` measured on a real Spark (Ubuntu 24.04,
driver 580.173, CUDA 13.0, Strata 0.2.7.1):

| Pack | Download | Decode tok/s | Prompt tok/s (4K-32K) |
|------|----------|--------------|------------------------|
| Qwen3.8-Flash-Next IQ3_S | 84 GB | 45.5-48.5 | 1,203-1,735 |
| Qwen3.8-Flash-Next NVFP4 | ~70 GB | 40.1-42.8 | 1,156-1,698 |

- Engine package: `linux-aarch64`, needs glibc ≥ 2.39 + driver ≥ 580.65.
- Six sm_121 perf switches auto-enable on GB10 (`STRATA_SM121_PERF=0` disables).
- `--mmap-experts` (auto on Spark): experts read in place from the model file —
  one copy in the unified pool instead of two.
- Not built for ARM upstream: desktop app, HIP/SYCL/CPU engines, Q2_0 and Gyro
  packs (their CPU kernels are x86-only — the engine refuses them at start).

## Abliterated model mapping (llmfit, GB10 profile)

Scored with `llmfit --profile gb10-spark` (128 GB unified, 273 GB/s). Every
existing category has a direct abliterated twin — swap `MODEL=` on the
category script rather than adding per-category abliterated scripts:

| Category | Aligned default | Abliterated twin (NVIDIA-servable) |
|----------|----------------|------------------------------------|
| chat | `cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` | `coughmedicine/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-W4A16` (44.6 GB, vLLM) or `-nvfp4` (48.3 GB) |
| code | `cyankiwi/Qwen3-Coder-Next-AWQ-4bit` | `gsting/Qwen3-Coder-Next-abliterated-FP8`, `sh0ck0r/Huihui-Qwen3-Coder-Next-abliterated-FP8` |
| reason | `codgician/...-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-int4` | `huihui-ai/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated` (BF16 ~70 GB) |
| general | `cyankiwi/Qwen3-Next-80B-A3B-Thinking-AWQ-4bit` | `amphineko/Huihui-Qwen3-Next-80B-A3B-Thinking-abliterated-AutoRound-W4A16` (41.5 GB) |
| lite | `Qwen/Qwen3-4B-Instruct-AWQ` | `huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated` (BF16 ~8 GB) |
| dedicated | `model-abliterated.sh` (port 8008) | `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (BF16 ~54 GB, SGLang — co-run headroom); max-quality alt: `Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF` swift-IQ3_S (~84 GB, Strata) |

The dedicated `model-abliterated.sh` exists because abliteration is a variant
of a role — one uncensored endpoint, not a fifth category axis. Aligned and
abliterated 80B-class models cannot co-run anyway (~90+ GB combined weights).
