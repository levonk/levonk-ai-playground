---
title: "FreeToken: Edge-Native MoE Serving with Bandwidth-Adaptive Execution"
date:
  created: 2026-08-26
  knowledge-basis: 2026-08-26
  last-used: 2026-08-26
tags:
  - research
  - inference-engine
  - moe
  - edge-serving
  - freetoken
  - flashml
  - adr-update-candidate
aliases:
  - FreeToken note
  - FlashML FreeToken
related:
  - "[[adr-202608021744-sglang-glom-runtime-mapping]]"
  - "[[runner-lib]]"
  - "[[model-reason]]"
source:
  - "https://github.com/FlashML-org/FreeToken"
  - "https://arxiv.org/abs/2608.16157"
  - "https://www.flashml.ai/"
---

# FreeToken: Edge-Native MoE Serving with Bandwidth-Adaptive Execution

> Research note — evaluates FreeToken as a new runtime for the DGX Spark tiered
> runtime model, and proposes an ADR update to
> [[adr-202608021744-sglang-glom-runtime-mapping]].

## TL;DR

**FreeToken** (FlashML, Apache 2.0) is an edge-native MoE serving engine that
treats a personal machine's GPU + CPU + host memory + interconnects as a
**unified, elastic inference platform**. It is not a replacement for SGLang or
vLLM across the board — it is a **new scenario class** that overlaps with both
the "frontier MoE that doesn't fit in VRAM" slot (currently Colibri in the ADR)
and the "in-VRAM MoE with agentic workloads" slot (currently SGLang). Matei
Zaharia (Databricks CTO) and Ion Stoica are coauthors on the paper
([arXiv:2608.16157](https://arxiv.org/abs/2608.16157), Aug 17 2026).

**Bottom line for this repo:** FreeToken should replace Colibri in the ADR's
tiered model and should also be evaluated as the default for MoE models that
benefit from its semantic-aware agentic caching — even when they fit in VRAM.
The user's current GPTQ-int4 distilled model stays on vLLM (FreeToken does not
support GPTQ), but the base Qwen3.5-35B-A3B can move to FreeToken with NVFP4.

## What FreeToken Is

| Field | Value |
|-------|-------|
| Project | [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) |
| Stars | 8.3k (as of 2026-08-26) |
| License | Apache 2.0 |
| Paper | [arXiv:2608.16157](https://arxiv.org/abs/2608.16157) — Aug 17 2026 |
| Authors | Shuo Yang, Xiaoze Fan, Melissa Pan, Haocheng Xi, Zhe Wang, Shanlin Sun, Kurt Keutzer, Song Han, **Matei Zaharia**, Chenfeng Xu, **Ion Stoica** |
| Website | [flashml.ai](https://www.flashml.ai/) |
| Install | `uv pip install "freetoken[accel]"` or desktop app |
| Inspired by | [mini-sglang](https://github.com/sgl-project/mini-sglang); reuses code from SGLang, vLLM, FlashInfer, flash-linear-attention, LightLLM, llama.cpp |

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed
for running frontier-scale open-weight models on personal and consumer
hardware. It treats heterogeneous edge resources — GPUs, CPUs, host memory, and
interconnects — as a unified, elastic inference platform.

## Key Innovations

### 1. Bandwidth-Adaptive CPU-GPU Co-Execution (q\* policy)

Rather than committing to a fixed offloading strategy (all-experts-on-GPU vs.
all-experts-on-CPU vs. disk-stream), FreeToken continuously maps computation
and model state onto the resources actually available. The `q*` policy
dynamically splits each step between PCIe fetch and CPU compute, overlapped.

Run `ft bench bw` once per machine to calibrate the split. The `auto` backend
resolves MoE models to `offload`, upgraded to `hybrid` when the cached
bandwidth profile recommends it.

**MoE backends:**
| Backend | Behavior |
|---------|----------|
| `fused` | Experts resident on GPU (needs the VRAM); never auto-selected |
| `offload` | Experts in host RAM, LRU cache of expert slots on GPU; misses stream over PCIe |
| `cpu` | Misses computed on CPU instead of fetched |
| `hybrid` | Per step: fetch some misses over PCIe, compute rest on CPU, overlapped |
| `auto` | Dense → `fused`; MoE → `offload`, upgraded to `hybrid` if `ft bench bw` recommends |

### 2. Semantic-Aware Caching (Agentic State Reuse)

This is the feature most directly relevant to this repo's multi-agent workload.
FreeToken features **semantic anchor checkpoints** for recurrent state and KV
caches, allowing agentic context edits (tool calls, thinking blocks) to avoid
redundant context recomputation.

When an agent edits its context mid-session (inserts a tool result, appends a
thinking block, truncates history), FreeToken identifies the semantic anchor
point and reuses the KV cache up to that point — only recomputing from the edit
forward. This is a significant advantage over SGLang/vLLM for multi-turn agent
loops where the system prompt + tool schema + early conversation are repeated
across every call.

### 3. Elastic Memory Management

Dynamic, runtime VRAM re-allocation between expert caches and KV memory
**without engine restarts or weight reloading**. As context grows, KV memory
expands; as expert cache hit rates shift, expert slots expand. No restart
needed.

### 4. FTW Fast Weight Format

`ft checkpoint` pre-converts a checkpoint into FreeToken's fast-load format.
`ft serve --model` auto-detects the result. Optional but recommended for
faster startup.

### 5. Broad MoE Support

Supports 20+ MoE models across various parameter scales and quantization
formats:

| Model | HF Checkpoints |
|-------|---------------|
| DeepSeek-V4 | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| GLM-5.2 (753B) | `nvidia/GLM-5.2-NVFP4` |
| GLM-4.7 | `nvidia/GLM-4.7-NVFP4` |
| **Qwen3.6 / Qwen3.5 MoE** | `Qwen/Qwen3.6-35B-A3B` (`-FP8`), `nvidia/Qwen3.6-35B-A3B-NVFP4`, `Qwen/Qwen3.5-35B-A3B` (`-FP8`) |
| Qwen3.6 dense | `Qwen/Qwen3.6-27B` (`-FP8`), `nvidia/Qwen3.6-27B-NVFP4` |
| Qwen3-MoE | `Qwen/Qwen3-30B-A3B` |
| gpt-oss | `openai/gpt-oss-120b`, `openai/gpt-oss-20b` |
| Gemma-4 | `google/gemma-4-26B-A4B-it`, `nvidia/Gemma-4-26B-A4B-NVFP4`, etc. |
| MiniMax-M2.5 | `nvidia/MiniMax-M2.5-NVFP4` |
| Muse-Glimmer | `meta-models/Muse-Glimmer-30B`, `RedHatAI/Muse-Glimmer-30B-NVFP4` |

**Supported quantization formats:** MXFP4, NVFP4, FP8, BF16, GGUF (for
Gemma-4). Loads HF safetensors directly.

**Not supported:** GPTQ (see compatibility section below).

### 6. Native Agent Integration

```bash
ft launch claude   # claude / codex / dsh / hermes / openclaw / opencode
```

Writes the agent's provider config, installs its CLI if missing, and starts it
against the FreeToken server. `--dry-run` previews the changes.

### 7. API Compatibility

- **OpenAI API**: `/v1/chat/completions`, `/v1/responses`, `/v1/models`
- **Anthropic API**: `/v1/messages`, `/v1/messages/count_tokens`

Any client library for either ecosystem works by pointing its base URL at the
server (default `127.0.0.1:1919`).

## Scale Claims (from the paper)

| Hardware | Model |
|----------|-------|
| 8GB laptop GPU | 35B MoE |
| Gaming desktop | 284B MoE |
| Single workstation GPU | 753B GLM-5.2 |

FreeToken "turns open weights into deployable local software, making the
machines users already own a practical platform for frontier-scale
intelligence."

## Relevance to the DGX Spark Setup

### Where FreeToken Fits in the Current ADR

The existing ADR
([[adr-202608021744-sglang-glom-runtime-mapping]]) defines a tiered runtime
selection:

| Scenario | Current Default | FreeToken Impact |
|----------|----------------|-----------------|
| In-VRAM MoE, multi-agent long-context | SGLang | **Contender** — semantic-aware caching is purpose-built for agent loops |
| In-VRAM dense models | SGLang | No impact — FreeToken is MoE-focused |
| GPTQ / compressed-tensors / multi-GPU PP | vLLM | No impact — FreeToken doesn't support GPTQ |
| Frontier MoE that doesn't fit in VRAM (GLM-5.2 753B) | **Colibri** | **Direct replacement** — FreeToken is more sophisticated, has agent integration, and is from a stronger team |
| Brand-new architecture / GGUF-only | llama.cpp | No impact — llama.cpp still has faster first-mover support |
| Code execution | Glom | No impact |

### Why FreeToken Replaces Colibri

1. **Same scenario class, better engine.** Both serve frontier MoE models that
   don't fit in VRAM. FreeToken's bandwidth-adaptive `hybrid` backend is a
   more sophisticated version of Colibri's disk-streaming — it dynamically
   splits between PCIe fetch and CPU compute based on a calibrated bandwidth
   profile, rather than purely streaming from disk.
2. **Agent-native.** Colibri is a C engine with an OpenAI-compatible API.
   FreeToken has semantic-aware caching for agentic context edits, native
   `ft launch` integration with Claude Code / Codex / OpenCode, and both
   OpenAI and Anthropic API compatibility. For this repo's multi-agent
   workload, FreeToken is purpose-built.
3. **Stronger team and trajectory.** Matei Zaharia, Ion Stoica, Song Han,
   Kurt Keutzer — the same Berkeley/Stanford lineage that produced SGLang,
   Spark, Databricks. 8.3k stars in ~9 days (paper submitted Aug 17, note
   written Aug 26). Colibri (JustVugg) is a smaller community project.
4. **Elastic memory.** FreeToken can dynamically re-allocate VRAM between
   expert caches and KV memory without restarts. Colibri cannot.
5. **Supports the model family we run.** FreeToken explicitly lists
   Qwen3.5-35B-A3B and Qwen3.6-35B-A3B as known-good checkpoints. Colibri
   was scoped to GLM-5.2 only.

### Why FreeToken May Also Augment SGLang for In-VRAM MoE

For MoE models that **do** fit in 128GB VRAM (like Qwen3.5-35B-A3B at ~22GB
INT4), FreeToken's `fused` backend runs experts resident on GPU — same as
SGLang. But FreeToken adds:

- **Semantic-aware caching**: agent tool calls and thinking blocks reuse KV
  cache instead of recomputing. This is a direct throughput win for the 32-128
  concurrent agent workload this repo targets.
- **Elastic memory**: as agent context grows, KV memory expands at the expense
  of expert cache — without restart. SGLang requires manual `--mem-fraction-static`
  tuning.
- **Bandwidth-adaptive fallback**: if a model doesn't quite fit in VRAM at the
  desired context length, FreeToken can transparently fall back to `hybrid`
  mode (some experts on CPU) instead of OOMing. SGLang has no such fallback.

**The trade-off:** SGLang has more production miles, broader model support
(dense models, non-MoE architectures), and the cu130-native container that
this repo already uses. FreeToken is MoE-focused and newer. The honest
split is: **SGLang for dense models and non-MoE architectures; FreeToken for
MoE models (both in-VRAM and frontier).**

## Compatibility with the Current Model

The user's current reasoning model is
`codgician/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-int4`,
served via vLLM on port 8002 ([[model-reason]]).

**FreeToken does not support GPTQ quantization.** Its supported formats are
MXFP4, NVFP4, FP8, BF16, and GGUF (Gemma-4 only). This means:

| Option | Path | Trade-off |
|--------|------|-----------|
| **A. Keep GPTQ model on vLLM** | `model-reason.sh` stays as-is (vLLM fallback) | No change; GPTQ-int4 is ~22GB, well-tested. But no semantic-aware caching for agent loops. |
| **B. Switch to base Qwen3.5-35B-A3B on FreeToken with NVFP4** | `ft serve --model nvidia/Qwen3.6-35B-A3B-NVFP4` or `Qwen/Qwen3.5-35B-A3B` | Gets FreeToken's agentic caching + elastic memory. Loses the Claude-4.6-Opus reasoning distillation. NVFP4 is a different quant than GPTQ-int4 — quality may differ. |
| **C. Use the unquantized Jackrong fine-tune on FreeToken** | `ft serve --model Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled` | Keeps the reasoning distillation. FreeToken handles quantization internally. Larger download (~70GB BF16 vs ~22GB GPTQ-int4). |
| **D. Re-quantize the distilled model to NVFP4** | Use NVIDIA's quantization tools to produce an NVFP4 checkpoint | Best of both worlds — distillation + FreeToken + NVFP4. Requires quantization work and validation. |

**Recommended path:** Start with **A** (keep GPTQ on vLLM) while evaluating
**B** (base model on FreeToken) in parallel on a different port. If
FreeToken's agentic caching delivers a measurable throughput win for the
multi-agent workload, pursue **D** (re-quantize the distilled model to NVFP4)
as the permanent migration.

## Proposed ADR Update

See the companion ADR amendment:
[[adr-20260826-freetoken-runtime-amendment]] — proposes adding FreeToken as
the default for MoE serving (both in-VRAM and frontier), replacing Colibri,
with SGLang retained for dense models and vLLM retained for GPTQ.

## Quick Start on DGX Spark

```bash
# Install
uv pip install "freetoken[accel]"

# Serve the base Qwen3.5-35B-A3B (NVFP4, auto backend selection)
ft serve --model nvidia/Qwen3.6-35B-A3B-NVFP4 --port 8004

# Or serve from a local checkpoint
ft serve --model ~/models/Qwen3.5-35B-A3B --port 8004

# Calibrate bandwidth (run once per machine)
ft bench bw

# Smoke-test
curl http://127.0.0.1:8004/v1/models
curl http://127.0.0.1:8004/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-35B-A3B",
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 256
  }'

# Launch a coding agent against it
ft launch claude --server http://127.0.0.1:8004
```

## Open Questions

- [ ] Does FreeToken's `hybrid` backend on the GB10 (128GB VRAM + 128GB host
      RAM) outperform SGLang's `fused` mode for Qwen3.5-35B-A3B at 64-128
      concurrent agents? Benchmark needed.
- [ ] Is the semantic-aware caching win measurable for Devin-style agent loops
      (long system prompt + tool schema + multi-turn)? Or does SGLang's
      RadixAttention already capture most of the prefix-reuse benefit?
- [ ] Can the Jackrong/Codgician Claude-4.6-Opus reasoning distillation be
      re-quantized to NVFP4 without quality loss? If yes, the GPTQ → NVFP4
      migration is clean.
- [ ] Does FreeToken support the GB10's SM103 / cu130 natively, or does it
      need the same CUDA version matching as SGLang? The README mentions RTX
      30/40/50 series — GB10 (Blackwell) is not explicitly listed.
- [ ] How does FreeToken's `ft launch` agent integration coexist with the
      existing Devin/Cursor/Continue setup? Does it conflict with provider
      configs?
- [ ] FreeToken's default port is 1919. This repo uses 8000-8003. Assign
      8004+ for FreeToken instances to stay within convention.

## References

- Paper: [FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution](https://arxiv.org/abs/2608.16157)
- Code: [https://github.com/FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)
- Website: [https://www.flashml.ai/](https://www.flashml.ai/)
- Quickstart: [docs/quickstart.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)
- Supported models: [docs/models.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- CLI reference: [docs/cli.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
- Install guide: [docs/install.md](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- Current ADR: [[adr-202608021744-sglang-glom-runtime-mapping]]
