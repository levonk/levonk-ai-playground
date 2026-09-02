---
modeline: "vim: set ft=markdown:"
title: "GDR: Tiered LLM Runtime Selection on DGX Spark — SGLang, vLLM, FreeToken, llama.cpp, Glom"
gdr-id: "gdr20260802001"
slug: "tiered-llm-runtime-selection"
url: "https://github.com/levonk/levonk-ai-playground/blob/main/internal-docs/adr/2026/08/adr-202608021744-sglang-glom-runtime-mapping.md"
synopsis: "Select the LLM runtime per scenario on DGX Spark: SGLang (cu130) as the default for supported in-VRAM dense models needing max throughput; FreeToken as the default for MoE models (both in-VRAM and frontier) with bandwidth-adaptive CPU-GPU co-execution and semantic-aware agentic caching; vLLM as the fallback for models/quants/parallelism SGLang and FreeToken do not support (GPTQ, compressed-tensors, multi-GPU pipeline parallel) or where production maturity matters; llama.cpp (ggml/GGUF) for brand-new architectures or GGUF-only quants not yet in SGLang/vLLM/FreeToken; Colibri retained as an emergency fallback only for frontier MoE if FreeToken has a critical bug on GB10; Glom only as a sandboxed code-execution engine for code-model outputs."
author: "https://github.com/levonk"
date-created: "2026-08-02"
date-updated: "2026-08-26"
date-review: "2027-02-02"
date-triggers: ["2026-11-02"]
version: "0.2.0"
status: "proposed"
aliases: []
tags: [doc/architecture/gdr]
supersedes: []
superseded-by: []
related-to: ["adr-20260826-freetoken-runtime-amendment", "adr-20260831-mac-studio-multi-host-deployment"]
scope:
  impact-scope:
    - "model-*.sh runner scripts"
    - "vllm-runner-lib.sh shared library"
    - "Docker base image (nvcr.io/nvidia/vllm → lmsysorg/sglang:latest-cu130, with vLLM retained as fallback)"
    - "FreeToken native Python install (uv pip install freetoken[accel]) — not Docker-based"
    - "Host port allocation per model (8000-8003 SGLang/vLLM, 8004+ FreeToken)"
    - "Multi-agent orchestration layer"
    - "MoE model serving (FreeToken default for both in-VRAM and frontier MoE)"
    - "Frontier MoE model serving (FreeToken hybrid backend; Colibri retained as emergency fallback)"
    - "llama.cpp (ggml/GGUF) bring-up path for brand-new architectures"
  excluded-scope:
    - "Model weights and quantization format selection (except GPTQ→NVFP4 migration evaluation)"
    - "Agent framework choice (Hermes/Paperclip/etc.)"
    - "Host OS and NVIDIA driver configuration"
hardware:
  target: "NVIDIA DGX Spark (Blackwell GB10 128GB / GB20)"
  container: "lmsysorg/sglang:latest-cu130 (SM103 / cu130 required) — default for dense models; vLLM retained as fallback"
  freetoken-install: "uv pip install freetoken[accel] — native Python, not Docker"
  validated-precisions: ["NVFP4", "FP8", "MXFP4", "AWQ-4bit", "GPTQ", "GGUF", "BF16"]
  freetoken-precisions: ["NVFP4", "FP8", "MXFP4", "BF16", "GGUF (Gemma-4 only)"]
  freetoken-unsupported: ["GPTQ", "AWQ (via GPTQ kernel)", "compressed-tensors"]
  moe-engine: "FreeToken (FlashML-org/FreeToken) — default for MoE serving (in-VRAM and frontier)"
  frontier-engine-fallback: "Colibri (JustVugg/colibri) — emergency fallback only if FreeToken has a critical bug on GB10"
  new-arch-engine: "llama.cpp (ggml/GGUF) — first-mover support for brand-new architectures and GGUF-only quants"
---

# Decision Record: Tiered LLM Runtime Selection on DGX Spark — SGLang, vLLM, FreeToken, llama.cpp, Glom

**Filename:** `adr-202608021744-sglang-glom-runtime-mapping.md`

- belongs in `internal-docs/adr/2026/08/`

> **Amended 2026-08-26 (v0.2.0):** FreeToken (FlashML, arXiv:2608.16157)
> added as the default runtime for MoE serving — both in-VRAM and frontier.
> Colibri demoted to emergency fallback. SGLang retained as default for dense
> models. vLLM retained for GPTQ and formats FreeToken does not support.
> See [`adr-20260826-freetoken-runtime-amendment.md`](adr-20260826-freetoken-runtime-amendment.md)
> for the full amendment rationale and the research note
> [`note-freetoken-edge-native-moe-serving.md`](../../../note-freetoken-edge-native-moe-serving.md).

---

## Context

This repo (`levonk-ai-playground`) currently deploys LLMs via vLLM in Docker
containers on a single NVIDIA GB10 128GB machine. Each `model-*.sh` script
pulls `nvcr.io/nvidia/vllm:26.04-py3` and starts an OpenAI-compatible vLLM
server on a unique host port. The audience is the operator running the GPU
box, running multi-agent workloads with long-context (64k–256k) inference and
high concurrency (32–128 concurrent agents).

The operator is evaluating the overall LLM runtime strategy for this box.
The questions are: (1) whether to switch the default inference runtime to
SGLang; (2) how to integrate a separate code-execution sandbox (Glom) into
the agent loop, since code models sit at the boundary between "generate
text" and "execute code"; (3) how to serve **huge frontier-level
Mixture-of-Experts (MoE) models** (e.g., GLM-5.2 at 744B parameters) that
physically cannot fit in the GB10's 128GB VRAM and therefore cannot be
served by either vLLM or SGLang on this hardware; (4) what to do for
**brand-new model architectures** or **GGUF-only quantizations** that
neither SGLang nor vLLM support yet; and (5) **whether FreeToken (FlashML)
should be the default for MoE serving** — both in-VRAM and frontier — given
its bandwidth-adaptive CPU-GPU co-execution, semantic-aware agentic
caching, and elastic memory management, which are purpose-built for the
multi-agent MoE workloads this repo targets.

Relevant background and prior art:

- NVIDIA's official **"SGLang for Inference | DGX Spark"** guide
  (https://build.nvidia.com/spark/sglang) — Spark-specific, uses the cu130
  container required for SM103, covers server mode + offline inference, and
  publishes a model support matrix for Spark-validated FP8/NVFP4 models.
- **DGX Spark Playbooks – SGLang**
  (https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/sglang/README.md)
  — engineering-level version of the NVIDIA guide with full Docker launch,
  CUDA/container-toolkit troubleshooting, and notes on context length,
  mem-fraction, and KV cache behavior.
- **Qwen3.6 + SGLang on DGX Spark (Weschera)**
  (https://github.com/Weschera/qwen-sglang-dgx-spark) — community guide
  tuned for real multi-agent workloads: SGLang v0.5.15-cu130 (first version
  with GB10-native Qwen NVFP4 support), 64 concurrent agents at 64k context,
  speculative decoding (NEXTN, Eagle top-k), and a reproducible SGLang vs
  vLLM comparison.
- **Colibri (JustVugg/colibri)** — a single-C-program engine that runs very
  large MoE models (e.g., GLM-5.2, 744B parameters) on a normal machine by
  streaming the model's experts from disk instead of needing them all in
  RAM/VRAM. CPU-only by default; GPU is optional. Per its quickstart
  (https://github.com/JustVugg/colibri/blob/main/docs/quickstart.md):
  ~16 GB RAM minimum (24 GB+ recommended), ~380 GB free disk for the int4
  model, a fast NVMe SSD is the single biggest factor in tokens/second
  (streaming speed = token speed), portable C with OpenMP and no x86-only
  intrinsics (builds unchanged on ARM64/Graviton), ships an OpenAI-compatible
  API + web dashboard, and is driven by a `coli` launcher
  (`COLI_MODEL=/nvme/glm52_i4 ./coli chat`). `--topp 0.85` reads fewer
  expert bytes per token with no quality loss, directly raising tokens/sec
  on a disk-bound machine. Placement only changes speed, never the model's
  answers or precision — it is still the full model. **As of the 2026-08-26
  amendment, Colibri is demoted to an emergency fallback only — FreeToken
  replaces it as the default for frontier MoE.**
- **FreeToken (FlashML-org/FreeToken)** — an edge-native MoE serving engine
  (Apache 2.0, https://github.com/FlashML-org/FreeToken) co-authored by
  Matei Zaharia and Ion Stoica (arXiv:2608.16157, Aug 17 2026). It treats
  heterogeneous edge resources — GPU, CPU, host memory, interconnects — as
  a unified, elastic inference platform. Core features: (1) bandwidth-
  adaptive CPU-GPU co-execution (`q*` policy) with `fused`/`offload`/`cpu`/
  `hybrid` MoE backends auto-selected per machine via `ft bench bw`
  calibration; (2) semantic-aware caching with anchor checkpoints for
  agentic context edits (tool calls, thinking blocks) that avoids redundant
  KV recomputation — purpose-built for multi-turn agent loops; (3) elastic
  VRAM re-allocation between expert caches and KV memory without engine
  restarts; (4) FTW fast weight format via `ft checkpoint`; (5) supports
  20+ MoE models (DeepSeek-V4, GLM-5.2, Qwen3.5/3.6-35B-A3B, gpt-oss,
  Gemma-4, MiniMax-M2.5, Muse-Glimmer) in MXFP4/NVFP4/FP8/BF16/GGUF; (6)
  native agent integration via `ft launch claude/codex/opencode`; (7)
  OpenAI + Anthropic API compatibility. Installed as a native Python
  package (`uv pip install freetoken[accel]`), not Docker. Does **not**
  support GPTQ. Inspired by mini-sglang; reuses code from SGLang, vLLM,
  FlashInfer, flash-linear-attention, LightLLM, and llama.cpp. 8.3k stars
  in ~9 days at the time of this amendment. See the research note
  [`note-freetoken-edge-native-moe-serving.md`](../../../note-freetoken-edge-native-moe-serving.md)
  for the full analysis.
- **vLLM (status quo)** — the existing runtime in this repo. Has more
  production miles than SGLang and often supports model architectures,
  attention variants, and quantization formats (GPTQ, compressed-tensors,
  some AWQ variants, FP8 layouts) before SGLang catches up. Stronger on
  multi-GPU pipeline parallelism and some LoRA-serving features. The
  known-good baseline when SGLang does not yet support a model or when
  production maturity matters more than throughput.
- **llama.cpp (ggml/GGUF)** — the ggml codebase is small and the community
  implements new model architectures fast, often before SGLang or vLLM
  merge support. If a model dropped last week, llama.cpp may be the only
  thing that runs it. Also the only runtime that reads GGUF quants
  (Q4_K_M, Q8_0, IQ4_XS, etc.) that the transformer-based servers do not
  read at all. CPU-first, single-file portable models (GGUF is one file),
  useful for quick bring-up before committing to a server runtime.
  llamafile, ollama, and koboldcpp are the same ggml family under different
  launchers.

## Constraints

- **Hardware is fixed**: NVIDIA DGX Spark with Blackwell GB10 128GB (and
  potentially GB20). The preferred container targets SM103/cu130, which
  SGLang's `lmsysorg/sglang:latest-cu130` does natively. vLLM's current
  `nvcr.io/nvidia/vllm:26.04-py3` image is not cu130-native and requires
  extra work to match — but vLLM is still retained as a fallback because it
  supports models/quants/parallelism that SGLang does not (see Decision).
- **One inference runtime per host port**: the existing `model-*.sh` pattern
  binds one model per host port. The decision must preserve this so multiple
  models can run simultaneously, even if different models use different
  runtimes.
- **Multi-agent, long-context workloads**: the operator runs 32–128
  concurrent agents at 64k–256k context. The runtime's scheduler, KV-cache
  reuse, and speculative decoding directly affect throughput.
- **Glom is not an inference server**: it is a code-execution sandbox. It
  must never be asked to load or serve an LLM.
- **Frontier MoE models do not fit in 128GB VRAM.** A 744B-parameter MoE
  model (e.g., GLM-5.2) at int4 is ~380GB on disk and cannot be resident in
  the GB10's VRAM. Neither vLLM nor SGLang can serve it on this hardware
  without an out-of-core path. FreeToken's `hybrid` backend (bandwidth-
  adaptive CPU-GPU co-execution) or Colibri's disk-streaming must be used.
  FreeToken is the default; Colibri is the emergency fallback.
- **Brand-new architectures and GGUF-only quants exist.** When a model
  architecture or quantization format is too new for SGLang, vLLM, and
  FreeToken, the operator still needs to run it. The runtime selection must
  include a first-mover path (llama.cpp/ggml) for these cases rather than
  blocking on upstream support.
- **FreeToken does not support GPTQ.** FreeToken supports MXFP4, NVFP4,
  FP8, BF16, and GGUF (Gemma-4 only). It does **not** support GPTQ, AWQ
  (via GPTQ kernel), or compressed-tensors. Models in those formats stay
  on vLLM until a re-quantization to a FreeToken-supported format (NVFP4
  recommended) is produced and validated.
- **FreeToken is native Python, not Docker.** Unlike SGLang and vLLM (which
  run in Docker containers), FreeToken is installed via
  `uv pip install freetoken[accel]` and runs directly on the host. This
  means it does not get Docker's resource isolation, but it also means it
  has direct access to both VRAM and host RAM for the hybrid backend — which
  is essential for its bandwidth-adaptive co-execution.
- **FreeToken GB10/SM103 support is unverified.** FreeToken's README lists
  RTX 30/40/50 series GPUs. The GB10 is Blackwell (SM103). Validation is
  required before FreeToken is accepted as the MoE default — if SM103 is not
  supported, FreeToken stays proposed-not-accepted and Colibri retains the
  frontier MoE slot.
- **No cross-ecosystem runtime mixing**: this repo maps within-ecosystem
  alternatives (e.g., pip→uv, npm→pnpm) but never across ecosystems. The
  same principle applies here — pick a default inference runtime per
  scenario class and document the fallbacks, rather than arbitrarily
  splitting models across runtimes. SGLang (dense), FreeToken (MoE), vLLM
  (GPTQ fallback), llama.cpp (new-arch), and Colibri (emergency frontier
  fallback) each own a distinct scenario class, not a competing slice of the
  same class.

## Decision

Select the LLM runtime **per scenario**, not per model type alone. SGLang
is the **default** for in-VRAM dense models; FreeToken is the **default**
for MoE models (both in-VRAM and frontier); vLLM, llama.cpp, and Colibri
are the documented fallbacks for the scenarios SGLang and FreeToken do not
cover. Glom is the only runtime for code execution.

The scenario-to-runtime mapping is:

| Scenario                                                      | Default runtime | Fallback / alternative                  |
|---------------------------------------------------------------|:---------------:|:---------------------------------------:|
| In-VRAM **dense** model, max throughput on Blackwell, multi-agent long-context | **SGLang (cu130)** | vLLM (if maturity/stability preferred) |
| In-VRAM **MoE** model, multi-agent long-context with tool calls / thinking blocks | **FreeToken** | SGLang (if FreeToken has issues on GB10) |
| Model/quant/parallelism SGLang and FreeToken do not support (GPTQ, compressed-tensors, multi-GPU pipeline parallel, some LoRA serving) | **vLLM** | SGLang/FreeToken (once support lands) |
| Brand-new architecture not yet in SGLang, vLLM, or FreeToken  | **llama.cpp (ggml)** | SGLang/vLLM/FreeToken (once upstream merges)     |
| GGUF-only quant (Q4_K_M, Q8_0, IQ4_XS, etc.)                  | **llama.cpp (ggml)** | Re-quantize to a server-supported format |
| Frontier MoE that does not fit in 128GB VRAM (e.g. GLM-5.2 744B) | **FreeToken** (hybrid backend) | Colibri (emergency fallback if FreeToken has a critical bug on GB10); Remote API (temporary; violates self-hosted) |
| Code execution (run code produced by a code model)            | **Glom** | — (no alternative; Glom is the only execution sandbox) |

Per model type, the defaults are:

| Model Type                  | FreeToken (MoE) | SGLang (dense) | vLLM (fallback) | llama.cpp (new-arch/GGUF) | Colibri (emergency) | Glom (execute) |
|-----------------------------|:---------------:|:--------------:|:---------------:|:-------------------------:|:-------------------:|:--------------:|
| General (dense)             | no              | default        | fallback        | if too new                | no                  | no             |
| Chat (dense)                | no              | default        | fallback        | if too new                | no                  | no             |
| Reasoning (MoE)             | default         | fallback       | if GPTQ         | no                        | no                  | no             |
| Code (MoE)                  | default (generate) | fallback (generate) | if GPTQ (generate) | no                   | no                  | yes (execute)  |
| Frontier MoE (e.g. GLM-5.2) | default         | no             | no              | no                        | emergency fallback  | no             |

In short: **SGLang is the default for dense models that fit in VRAM;
FreeToken is the default for MoE models (both in-VRAM and frontier);
vLLM is the fallback when SGLang/FreeToken cannot serve it (especially
GPTQ); llama.cpp is the first-mover for what is too new for any server
runtime; Colibri is the emergency fallback for frontier MoE if FreeToken
fails on GB10; Glom runs the code a code model produces.** No runtime is
asked to do another's job, and no model is left without a serving path.

## Rationale

**Why SGLang is the default for in-VRAM dense models on DGX Spark (not the
only option):**

- **GB10/GB20-native container**: SGLang ships a `latest-cu130` image built
  for SM103, which is the correct CUDA target for Blackwell. vLLM's current
  pinned image (`nvcr.io/nvidia/vllm:26.04-py3`) is not cu130-native and
  requires extra work to match — so SGLang is the default where both can
  serve the same model, but vLLM is retained for the cases below.
- **Higher throughput on Blackwell for supported models**: the Weschera
  benchmark shows SGLang sustaining 64 concurrent agents at 64k context on
  a single Spark, with speculative decoding (NEXTN, Eagle top-k) that vLLM
  does not expose at parity on this hardware.
- **NVFP4 + FP8 support**: SGLang v0.5.15-cu130 is the first version with
  GB10-native Qwen NVFP4 support, matching the AWQ-4bit models this repo
  already runs.
- **Long-context performance**: reasoning models (DeepSeek-R1, QwQ,
  Nemotron-R) need 64k–256k context and large KV footprints. SGLang's
  router + scheduler + KV-cache reuse are built for this.

**Why vLLM is retained as the fallback (not abandoned):**

- **Broader/stable architecture support**: vLLM has more production miles
  and often supports model architectures and attention variants before
  SGLang or FreeToken catches up. When a model is not in SGLang's or
  FreeToken's supported matrix, vLLM is the answer, not a workaround.
- **Quantization format coverage**: GPTQ, some AWQ variants,
  compressed-tensors, and FP8 layouts where vLLM's kernels are more
  mature. SGLang's NVFP4 path on GB10 is a real advantage, but it is not
  universal across every quant. FreeToken does not support GPTQ at all —
  so the user's current
  `Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-int4` stays
  on vLLM until a NVFP4 re-quantization is produced and validated.
- **Multi-GPU pipeline parallelism and some LoRA-serving features** where
  vLLM is ahead.
- **Production maturity**: when the operator needs the known-good baseline,
  not the throughput frontier, vLLM is the right choice. The decision is
  "SGLang/FreeToken by default, vLLM when they cannot serve it or
  stability wins," not "SGLang/FreeToken only."

**Why FreeToken is the default for MoE serving (both in-VRAM and frontier):**

- **Semantic-aware agentic caching**: FreeToken's anchor checkpoints for
  agentic context edits (tool calls, thinking blocks) avoid redundant KV
  recomputation in multi-turn agent loops. This is purpose-built for the
  32–128 concurrent agent workload this repo targets, where the system
  prompt + tool schema + early conversation are repeated across every call.
  SGLang's RadixAttention captures prefix reuse but does not handle mid-
  session context edits the same way.
- **Bandwidth-adaptive CPU-GPU co-execution**: FreeToken's `q*` policy
  dynamically splits each step between PCIe fetch and CPU compute based on
  a calibrated bandwidth profile (`ft bench bw`). For in-VRAM MoE models,
  the `fused` backend runs experts resident on GPU (same as SGLang). For
  frontier MoE that doesn't fit, the `hybrid` backend transparently falls
  back to a mix of PCIe fetch and CPU compute — no OOM cliff.
- **Elastic memory management**: dynamic, runtime VRAM re-allocation
  between expert caches and KV memory without engine restarts or weight
  reloading. As agent context grows, KV memory expands; as expert cache
  hit rates shift, expert slots expand. SGLang requires manual
  `--mem-fraction-static` tuning.
- **No OOM cliff for MoE**: if a MoE model doesn't quite fit in VRAM at
  the desired context length, FreeToken transparently falls back to
  `hybrid` mode (some experts on CPU) instead of crashing. SGLang and vLLM
  have no such fallback — they OOM.
- **Native agent integration**: `ft launch claude/codex/opencode` writes
  the agent's provider config, installs its CLI if missing, and starts it
  against the FreeToken server. Both OpenAI and Anthropic API
  compatibility. No equivalent in SGLang, vLLM, or Colibri.
- **Broad MoE support**: 20+ MoE models including DeepSeek-V4, GLM-5.2,
  Qwen3.5/3.6-35B-A3B, gpt-oss, Gemma-4, MiniMax-M2.5, Muse-Glimmer.
  Directly covers the model families this repo runs.
- **Strong team and trajectory**: co-authored by Matei Zaharia (Databricks
  CTO), Ion Stoica, Song Han, Kurt Keutzer — the same Berkeley/Stanford
  lineage that produced SGLang, Spark, and Databricks. 8.3k GitHub stars
  in ~9 days. Inspired by mini-sglang; reuses code from SGLang, vLLM,
  FlashInfer, flash-linear-attention, LightLLM, and llama.cpp.
- **Limitation — not for dense models**: FreeToken is MoE-focused. For
  dense models (Qwen3-Next-80B, etc.), SGLang remains the default due to
  broader architecture support and the cu130-native container.
- **Limitation — no GPTQ**: FreeToken supports MXFP4, NVFP4, FP8, BF16,
  and GGUF (Gemma-4 only). It does **not** support GPTQ. GPTQ models stay
  on vLLM.
- **Limitation — very new**: paper submitted Aug 17 2026; ~9 days old at
  the time of this amendment. Production maturity is unknown. SGLang has
  400k+ GPUs in production; FreeToken has no documented production
  deployments yet. SGLang is retained as the documented fallback for
  in-VRAM MoE if FreeToken has issues on GB10.

**Why llama.cpp (ggml/GGUF) is the first-mover path for brand-new
architectures and GGUF-only quants:**

- **First-mover architecture support**: the ggml codebase is small and the
  community implements new model architectures fast, often before SGLang,
  vLLM, or FreeToken merge support. If a model dropped last week, llama.cpp
  may be the only thing that runs it.
- **GGUF-only quantizations**: GGUF quants (Q4_K_M, Q8_0, IQ4_XS, etc.)
  are not read by the transformer-based servers at all. llama.cpp is the
  only runtime that reads them. (FreeToken reads GGUF for Gemma-4 only,
  not broadly.)
- **CPU-first / single-file portable**: GGUF is one file; useful for quick
  bring-up before committing to a server runtime. llamafile, ollama, and
  koboldcpp are the same ggml family under different launchers.
- **Not a replacement for SGLang/FreeToken/vLLM on supported models**:
  llama.cpp lacks the batching, KV-cache reuse, and multi-agent concurrency
  of the server runtimes. It is the bring-up/new-arch path, not the
  production serving path for models the servers already support.

**Why Glom is scoped to execution only:**

- Glom is a code-execution engine, not an inference runtime. Asking it to
  serve general/chat/reasoning models would misuse it and lose batching,
  KV-cache reuse, and concurrency.
- The only place Glom adds value is **executing** the code that a
  code model produces — sandboxed Python/JS/shell evaluation and tool
  calls, regardless of whether the code model was served by SGLang, vLLM,
  or llama.cpp.

**Why Colibri is demoted to emergency fallback (no longer the frontier MoE
default):**

- **FreeToken replaces Colibri as the frontier MoE default.** FreeToken's
  bandwidth-adaptive `hybrid` backend is a more sophisticated version of
  Colibri's disk-streaming — it dynamically splits between PCIe fetch and
  CPU compute based on a calibrated bandwidth profile, rather than purely
  streaming from disk. FreeToken also adds semantic-aware caching, elastic
  memory, native agent integration, and supports 20+ MoE models (Colibri
  was scoped to GLM-5.2 only).
- **Colibri is retained as an emergency fallback only** — if FreeToken has
  a critical bug on the GB10 (SM103 support is unverified), Colibri's
  disk-streaming path remains available. Colibri's `coli doctor` and
  `coli plan` health checks, OpenAI-compatible API, and `--topp 0.85`
  optimization are unchanged.
- A 744B-parameter MoE model at int4 is ~380GB on disk and physically
  cannot be resident in the GB10's 128GB VRAM. Neither vLLM nor SGLang can
  serve it on this hardware without an out-of-core path. FreeToken's
  `hybrid` backend or Colibri's disk-streaming must be used. FreeToken is
  the default; Colibri is the emergency fallback.
- Colibri streams the model's experts from NVMe on demand, so streaming
  speed = token speed. A fast NVMe SSD is the single biggest factor in
  tokens/sec, and placement only changes speed — never the model's answers
  or precision. It is still the full model.
- It is a single C program with OpenMP and no x86-only intrinsics (builds
  unchanged on ARM64/Graviton), ships an OpenAI-compatible API + web
  dashboard, and is driven by a `coli` launcher.
- It is **not** a replacement for SGLang/FreeToken/vLLM on in-VRAM models.
  For models that fit in 128GB VRAM, the server runtimes' batching,
  KV-cache reuse, speculative decoding, and GB10-native cu130 container
  beat a CPU-default, disk-streamed engine on throughput and latency.

**Why a tiered default-with-fallbacks model instead of one runtime for
everything:**

- No single runtime covers every scenario. SGLang wins on throughput for
  supported dense models on Blackwell but does not support every
  model/quant; FreeToken wins for MoE with agentic caching and bandwidth-
  adaptive execution but is MoE-only and does not support GPTQ; vLLM has
  broader support but is not cu130-native and is slower on this hardware;
  llama.cpp is the only path for brand-new architectures and GGUF quants
  but lacks server-grade concurrency; Colibri is the emergency fallback for
  frontier MoE if FreeToken fails. Pretending any one of them is the only
  runtime leaves real models without a serving path.
- The tiered model keeps one **default** per scenario class (so the
  operator is not making a fresh decision every time) while documenting
  the **fallback** (so the operator is never blocked).

**Trade-offs and risks:**

- **Migration cost**: the default path moves `model-*.sh` scripts and the
  shared library from vLLM to SGLang (dense) and FreeToken (MoE). The
  shared library's vLLM-specific flags (`--quantization awq_merlin`,
  `--max-model-len`, speculative decoding flags) need SGLang equivalents.
  FreeToken uses a different launch path (native Python, not Docker) so
  the library needs a new `run_freetoken()` function. vLLM scripts are
  retained (not deleted) as the fallback path.
- **Container image change (SGLang path)**: `nvcr.io/nvidia/vllm:26.04-py3`
  → `lmsysorg/sglang:latest-cu130`. The `latest` tag is a floating
  reference; pinning to a specific version (e.g., `v0.5.15-cu130`) is
  required for reproducibility. vLLM's image is retained for fallback
  models.
- **FreeToken is native Python, not Docker**: FreeToken runs outside the
  container isolation boundary. For a single-operator playground this is
  acceptable, but it means FreeToken does not get Docker's resource
  isolation. It also means FreeToken has direct access to both VRAM and
  host RAM, which is essential for its hybrid backend.
- **FreeToken GB10/SM103 support is unverified**: FreeToken's README lists
  RTX 30/40/50 series. The GB10 is Blackwell (SM103). If SM103 is not
  supported, FreeToken stays proposed-not-accepted and Colibri retains the
  frontier MoE slot. This is the single biggest validation gate.
- **FreeToken is very new** (paper Aug 17 2026, ~9 days old at amendment
  time). Production maturity is unknown. SGLang is retained as the
  documented fallback for in-VRAM MoE.
- **More operational surface**: the operator now manages up to five
  runtimes (SGLang dense, FreeToken MoE, vLLM fallback, llama.cpp bring-up,
  Colibri emergency) plus Glom. Each has its own pin discipline and health
  checks. This is the honest cost of covering every scenario instead of
  pretending one runtime suffices. FreeToken replaces Colibri as the
  frontier MoE default, so the net runtime count is the same as before the
  amendment (Colibri was already counted).
- **Glom is a new dependency**: it must be installed, networked to the
  agent loop, and kept isolated from the inference runtimes.

## Technical Approach

### Container and launch (dense default path: SGLang)

For models moving to the SGLang default, replace the vLLM base image with
the SGLang cu130 image:

```bash
docker pull lmsysorg/sglang:latest-cu130
docker run --rm --gpus all lmsysorg/sglang:latest-cu130 nvidia-smi
```

Each `model-*.sh` script continues to bind a unique host port, but launches
the SGLang server instead of the vLLM server. The shared library
(`vllm-runner-lib.sh` → rename to `sglang-runner-lib.sh`) provides the
common launch function; per-model scripts set `MODEL`, `HOST_PORT`,
quantization, context length, and speculative-decoding flags.

### Fallback path: vLLM retained

Models that SGLang and FreeToken do not support (or where vLLM's
stability/quant coverage is preferred) keep their existing `model-*.sh`
launching `nvcr.io/nvidia/vllm:26.04-py3` (or a cu130-native vLLM image
once one ships). The shared library retains a vLLM launch function
alongside the SGLang and FreeToken ones. The decision of which runtime a
given model uses is encoded in the model script (or a per-model config),
not made at runtime.

### MoE default path: FreeToken (native Python)

For MoE models (both in-VRAM and frontier), FreeToken is the default
runtime. Unlike SGLang and vLLM, FreeToken is installed as a native
Python package and runs directly on the host — not in Docker:

```bash
# Install
uv pip install "freetoken[accel]"

# Calibrate bandwidth (run once per machine)
ft bench bw

# Serve an in-VRAM MoE model (e.g., Qwen3.5-35B-A3B NVFP4)
ft serve --model nvidia/Qwen3.6-35B-A3B-NVFP4 --port 8004

# Serve a frontier MoE model (e.g., GLM-5.2 753B NVFP4)
ft serve --model nvidia/GLM-5.2-NVFP4 --port 8005 --moe-backend hybrid

# Smoke-test
curl http://127.0.0.1:8004/v1/models
curl http://127.0.0.1:8004/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-35B-A3B","messages":[{"role":"user","content":"ping"}],"max_tokens":256}'

# Launch a coding agent against the server
ft launch claude --server http://127.0.0.1:8004
```

The shared library (`runner-lib.sh`) needs a new `run_freetoken()`
function that does **not** use `docker run` — it launches `ft serve`
directly on the host. FreeToken instances use host ports 8004+ to stay
within the repo's port convention (8000-8003 are taken by SGLang/vLLM
scripts). FreeToken's default port is 1919; override with `--port 8004`.

New `model-freetoken-*.sh` scripts follow the existing `model-*.sh`
pattern but set `RUNTIME=freetoken` and call `run_freetoken()` instead of
`run_container()`.

### Bring-up path: llama.cpp (ggml/GGUF)

For brand-new architectures or GGUF-only quants not yet in
SGLang/vLLM/FreeToken, use llama.cpp (or a ggml-family launcher:
llamafile, ollama, koboldcpp) as the bring-up runtime:

```bash
# Build llama.cpp with CUDA support for the GB10
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release

# Serve a GGUF model with the OpenAI-compatible server
./build/bin/llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 80XX
```

This is the first-mover path: once SGLang, vLLM, or FreeToken merges
support for the architecture/quant, migrate the model off llama.cpp to
the server runtime for batching and multi-agent concurrency. llama.cpp is
not the production serving path for models the servers already support.

### Model-to-runtime routing in the agent loop

- "Write code" (MoE) → code model on its host port (FreeToken default; SGLang fallback; vLLM if GPTQ; llama.cpp if too new)
- "Write code" (dense) → code model on its host port (SGLang default; vLLM fallback; llama.cpp if too new)
- "Run code"   → Glom (sandboxed execution of code-model output, regardless of which runtime served it)
- "Think deeply" (MoE) → reasoning model (FreeToken default; SGLang fallback; vLLM if GPTQ)
- "Think deeply" (dense) → reasoning model (SGLang default; vLLM fallback)
- "Chat" / "General tasks" (dense) → general/chat model (SGLang default; vLLM fallback)
- "Frontier MoE / huge model" → FreeToken (hybrid backend, OpenAI + Anthropic API); Colibri emergency fallback
- "Brand-new architecture / GGUF-only quant" → llama.cpp (bring-up; migrate to SGLang/vLLM/FreeToken once supported)

The agent framework (Hermes/Paperclip/etc.) routes outputs from the code
model into Glom for execution, and feeds execution results back into the
next inference call. Frontier MoE calls go to FreeToken's OpenAI +
Anthropic-compatible endpoint (or Colibri's OpenAI-compatible endpoint as
the emergency fallback); bring-up calls go to llama.cpp's OpenAI-compatible
server. From the agent's perspective, every runtime exposes the same
OpenAI API shape on a different host port, so routing is URL config, not
framework changes.

### Frontier MoE serving with FreeToken (default)

FreeToken's `hybrid` backend is the default for frontier MoE models that
do not fit in 128GB VRAM (e.g., GLM-5.2 753B). Run `ft bench bw` once to
calibrate the bandwidth-adaptive split for the GB10's VRAM-to-host-RAM
bandwidth, then serve with `--moe-backend hybrid`:

```bash
ft bench bw                                          # calibrate (once per machine)
ft serve --model nvidia/GLM-5.2-NVFP4 --port 8005 --moe-backend hybrid
```

FreeToken dynamically splits each step between PCIe fetch and CPU compute,
overlapped. The calibrated profile tells it the optimal split for this
specific hardware. Semantic-aware caching and elastic memory management
work the same as for in-VRAM MoE — the agent loop gets the same benefits.

### Frontier MoE emergency fallback: Colibri

Colibri runs outside the SGLang Docker stack — it is a native C engine on
the host, CPU-default with optional GPU. Per the quickstart:

```bash
# Build from source (fastest binary for this CPU; ARCH=native unlocks host vector instructions)
sudo apt install -y build-essential git python3
git clone https://github.com/JustVugg/colibri.git && cd colibri && ./setup.sh

# Or use the prebuilt archive (Linux x86_64 / macOS / Windows)
mkdir colibri && tar xzf colibri-v1.1.0-linux-x86_64.tar.gz -C colibri && cd colibri
python3 coli info        # engine ready

# Serve a frontier MoE model (e.g. GLM-5.2 744B int4, ~380GB on NVMe)
COLI_MODEL=/nvme/glm52_i4 ./coli chat --topp 0.85
COLI_MODEL=/nvme/glm52_i4 ./coli doctor   # read-only readiness check
COLI_MODEL=/nvme/glm52_i4 ./coli plan     # shows where the model lives (RAM/disk/GPU)
```

Operational notes from the quickstart that affect this ADR:

- **Disk is the bottleneck.** A fast NVMe SSD is the single biggest factor
  in tokens/sec; on a slow or shared disk, generation can be well under 1
  token/sec. The GPU box must reserve a dedicated, fast NVMe path for the
  ~380GB int4 model files.
- **`--topp 0.85` is a free win on disk-bound machines** — it reads fewer
  expert bytes per token with no quality loss, directly raising
  tokens/sec. Default it on for frontier MoE serving.
- **First launch loads the resident weights (~10 GB)** — expect a startup
  pause; do not treat it as a hang.
- **Placement only changes speed, never answers or precision.** Frontier
  MoE output is the full model's output, not a degraded approximation.
- **OpenAI-compatible API + web dashboard** (see `docs/api.md`) — the
  agent loop can treat Colibri as just another OpenAI endpoint on a new
  host port, so no agent-framework changes are needed beyond a URL config.
- **`libgomp.so.1` runtime dependency** — on minimal cloud images or fresh
  containers that have never had a compiler, install `libgomp1` so the
  engine does not exit silently at startup. `coli doctor` names the
  missing library.

### Image and engine pinning

Pin the SGLang image to a specific version tag rather than `latest` to keep
deploys reproducible. Initial recommendation: `lmsysorg/sglang:v0.5.15-cu130`
(first version with GB10-native Qwen NVFP4 support, per the Weschera guide).

Retain the vLLM fallback image pinned at `nvcr.io/nvidia/vllm:26.04-py3`
(or a cu130-native vLLM image once one ships) for models that stay on vLLM
(especially GPTQ models that FreeToken does not support).

Pin FreeToken to a specific PyPI version or git tag rather than installing
`freetoken[accel]` unpinned. FreeToken is very new (paper Aug 17 2026) and
the API may change; pinning prevents a bad upstream release from breaking
the GPU box. Record the version in the model script or a `.freetoken-version`
file. Run `ft bench bw` after each version bump to re-calibrate the
bandwidth profile.

Pin llama.cpp to a recorded commit hash when building from source
(`-DGGML_CUDA=ON` for the GB10), or to a specific release tag. Bring-up
models on llama.cpp should be migrated to SGLang/vLLM/FreeToken once
upstream support lands, at which point the llama.cpp pin for that model
is retired.

Pin Colibri to a specific release archive (e.g.,
`colibri-v1.1.0-linux-x86_64.tar.gz`) rather than building from `main`,
and record the commit hash if building from source with `ARCH=native`.
Colibri's `coli doctor` should pass before the emergency fallback endpoint
is advertised to the agent loop. Colibri is now the emergency fallback
only — FreeToken is the default for frontier MoE.

## Affected Components

- **`vllm-runner-lib.sh`** — refactored into a multi-runtime shared library
  (`runner-lib.sh` or `sglang-runner-lib.sh` with retained vLLM and new
  FreeToken launch functions); vLLM-specific flags kept for fallback
  models, SGLang flags added for dense default-path models, FreeToken
  launch path added for MoE default-path models.
- **`model-chat.sh`, `model-code.sh`, `model-reasoning.sh`, `model-general.sh`**
  — each migrated to its default runtime (SGLang for dense, FreeToken for
  MoE, vLLM for GPTQ) on its existing host port (8000–8003). A per-model
  config encodes which runtime each model uses.
- **New `model-freetoken-*.sh` scripts** — one per MoE model served by
  FreeToken. Initial set: `model-freetoken-reason.sh` (Qwen3.5-35B-A3B
  NVFP4, port 8004), `model-freetoken-frontier.sh` (GLM-5.2 NVFP4, port
  8005). These set `RUNTIME=freetoken` and call `run_freetoken()`.
- **Docker base images** — `nvcr.io/nvidia/vllm:26.04-py3` retained for
  fallback models; `lmsysorg/sglang:<pinned>-cu130` added for dense
  default-path models. FreeToken does not use Docker (native Python).
- **llama.cpp bring-up path** — a new `model-bringup-*.sh` pattern (or a
  `llama-runner-lib.sh` function) for brand-new architectures / GGUF-only
  quants, using `llama-server` on a dedicated host port.
- **README model table** — gains a "Runtime" column (SGLang / FreeToken /
  vLLM / llama.cpp / Colibri) alongside the existing "Special Config"
  column; the "Special Config" column moves from vLLM-only flags to
  per-runtime flags; new rows for FreeToken MoE, frontier MoE (FreeToken
  default, Colibri emergency fallback), and bring-up (llama.cpp).
- **`.env` / launch env vars** — `HOST_PORT`, `MODEL`, `ACCT`,
  `GPU_MEMORY_UTILIZATION` semantics may shift per runtime; FreeToken adds
  `--moe-backend` and `ft bench bw` calibration; Colibri adds
  `COLI_MODEL` (path to the on-disk int4 model) and `--topp 0.85` as a
  default for disk-bound frontier serving; llama.cpp adds `GGUF_PATH`
  and `llama-server` flags.
- **Agent loop / orchestrator** — gains a Glom execution step for code
  model outputs, a FreeToken endpoint config for MoE calls (with Colibri
  as emergency fallback), and a llama.cpp endpoint config for bring-up
  calls (all OpenAI-compatible, so URL-only). FreeToken's `ft launch`
  can auto-configure Claude Code / Codex / OpenCode.
- **NVMe storage layout** — a dedicated, fast NVMe path must be reserved
  for the ~380GB int4 frontier MoE model files; disk speed = token speed
  (for Colibri emergency fallback). FreeToken's hybrid backend uses host
  RAM + PCIe, so NVMe speed matters less than for Colibri but still
  affects model loading.
- **Operator runbook** — startup, smoke-test, and troubleshooting steps
  per runtime (SGLang dense, FreeToken MoE, vLLM fallback, llama.cpp
  bring-up, Colibri emergency); a decision tree for "which runtime does
  this model use?"

## Consequences

### Negative

- **Migration cost for the default path**: the `model-*.sh` scripts moving
  to SGLang (dense) and FreeToken (MoE) and the shared library must be
  rewritten for their launch flags. No incremental vLLM→SGLang/FreeToken
  shim; the launch flags and server API differ enough that a clean
  rewrite is cheaper than a compatibility layer. vLLM scripts are
  retained (not deleted) for fallback models, so the library carries all
  three launch functions.
- **More operational surface**: the operator now manages up to five
  runtimes (SGLang dense, FreeToken MoE, vLLM fallback, llama.cpp
  bring-up, Colibri emergency) plus Glom. Each has its own pin discipline
  and health checks. This is the honest cost of covering every scenario
  instead of pretending one runtime suffices. FreeToken replaces Colibri
  as the frontier MoE default, so the net runtime count is the same as
  before the amendment.
- **FreeToken is native Python, not Docker** — it runs outside the
  container isolation boundary. For a single-operator playground this is
  acceptable, but it means FreeToken does not get Docker's resource
  isolation. It also means FreeToken has direct access to both VRAM and
  host RAM, which is essential for its hybrid backend.
- **FreeToken GB10/SM103 support is unverified** — the single biggest
  validation gate. If SM103 is not supported, FreeToken stays
  proposed-not-accepted and Colibri retains the frontier MoE slot.
- **FreeToken is very new** (paper Aug 17 2026, ~9 days old at amendment
  time). Production maturity is unknown. SGLang is retained as the
  documented fallback for in-VRAM MoE.
- **Colibri is a native C engine on the host (outside Docker)**, with its
  own build/pin discipline, `libgomp1` runtime dep, and
  `coli doctor`/`coli plan` health checks. Retained as emergency fallback
  only.
- **llama.cpp bring-up models must be migrated** to SGLang/vLLM/FreeToken
  once upstream support lands, or they accumulate as technical debt on a
  non-production runtime.
- **NVMe disk is the frontier-MoE bottleneck (Colibri fallback).**
  Tokens/sec is bounded by streaming speed; on a slow or shared disk,
  generation can be well under 1 token/sec. FreeToken's hybrid backend
  is less disk-bound (uses host RAM + PCIe), but the NVMe path still
  matters for model loading.
- **Re-validation of every model**: each `model-*.sh` moving to SGLang or
  FreeToken must be re-benchmarked to confirm the throughput and
  context-length claims hold for the specific weights this repo uses;
  fallback models stay on vLLM and are re-validated against the latest
  vLLM release; frontier MoE must be benchmarked on FreeToken (hybrid
  backend) separately, with `ft bench bw` confirming the bandwidth
  profile; bring-up models on llama.cpp are validated only enough to
  confirm correctness, not throughput.
- **Image and engine pinning discipline required**: `latest-cu130` is a
  floating tag, FreeToken's PyPI package moves fast, Colibri's `main`
  branch moves, and llama.cpp commits move fast. Without pinning, a bad
  upstream push can break the GPU box on next pull/build.

### Positive

- **Higher throughput on the same hardware for supported dense models**:
  SGLang's scheduler, KV-cache reuse, and speculative decoding are tuned
  for Blackwell and the multi-agent, long-context workloads this repo
  targets.
- **Higher throughput for MoE models with agentic workloads**: FreeToken's
  semantic-aware caching, bandwidth-adaptive co-execution, and elastic
  memory are purpose-built for the 32–128 concurrent agent MoE workload
  this repo targets.
- **Correct CUDA target by default**: cu130 / SM103 is the right container
  for GB10/GB20, removing the manual cu130 workaround vLLM requires for
  SGLang default-path models.
- **No model is left without a serving path.** SGLang covers supported
  in-VRAM dense models; FreeToken covers MoE models (in-VRAM and
  frontier); vLLM covers the models/quant/parallelism SGLang and
  FreeToken do not (especially GPTQ); llama.cpp covers brand-new
  architectures and GGUF-only quants; Colibri is the emergency fallback
  for frontier MoE. The tiered model is honest about the fact that no
  single runtime covers everything.
- **Clean separation of concerns**: SGLang (dense) / FreeToken (MoE) /
  vLLM (GPTQ fallback) / llama.cpp (new-arch) own inference (each for
  their scenario); Glom owns code execution; Colibri owns emergency
  disk-streamed frontier MoE. Non-overlapping scenario classes.
- **NVFP4 support**: SGLang v0.5.15-cu130 and FreeToken both add GB10-
  native Qwen NVFP4, opening a precision path vLLM does not expose on
  this hardware.
- **Frontier MoE is now servable on this hardware at all.** Without
  FreeToken or Colibri, a 744B MoE model is simply out of reach on a
  128GB-VRAM box. FreeToken's hybrid backend makes it possible with
  bandwidth-adaptive CPU-GPU co-execution, and its OpenAI + Anthropic
  API compatibility means the agent loop treats it as just another
  endpoint. Colibri remains as the emergency fallback.
- **No OOM cliff for MoE**: FreeToken transparently falls back to
  `hybrid` mode if a MoE model doesn't quite fit in VRAM at the desired
  context length. SGLang and vLLM have no such fallback — they OOM.
- **Native agent integration**: FreeToken's `ft launch` auto-configures
  Claude Code, Codex, and OpenCode against the server. No equivalent in
  SGLang, vLLM, or Colibri.
- **Brand-new architectures are servable on day one** via llama.cpp,
  instead of blocking on SGLang/vLLM/FreeToken upstream merges.

### Neutral

- **Host port allocation is unchanged for existing scripts**: each
  in-VRAM model keeps its unique port (8000–8003), so the multi-model-
  simultaneous-run property is preserved. FreeToken instances get new
  host ports (8004+), Colibri (emergency) and llama.cpp bring-up each
  get new host ports for their OpenAI-compatible endpoints, extending
  the same pattern.
- **`.env` and `HUGGING_FACE_HUB_TOKEN` handling is unchanged**: SGLang,
  vLLM, and FreeToken all read the same HF token for private model
  pulls. Colibri reads model files from `COLI_MODEL` on disk, so it
  does not need the HF token at serve time (only at model-prep time,
  which is a one-time Python step). llama.cpp reads GGUF files from
  disk, same as Colibri.

## Alternatives Considered

**Option A — Stay on vLLM (status quo).**
**Option A — Stay on vLLM only (status quo).**
Pros: zero migration cost; existing scripts and shared library keep working;
vLLM is well-understood by the operator. Cons: no cu130-native container for
SM103 on the default path; lower throughput on Blackwell for multi-agent
long-context workloads that SGLang handles better; no GB10-native NVFP4
path; misses speculative decoding parity on this hardware; no path for
brand-new architectures or GGUF-only quants; no path for frontier MoE.

**Option B — SGLang only for all in-VRAM inference (the earlier draft of
this ADR).**
Pros: one container base, one set of flags, simplest mental model. Cons:
leaves models/quant/parallelism SGLang does not support without a serving
path; leaves brand-new architectures and GGUF-only quants without a serving
path; pretends a single runtime covers every in-VRAM scenario when it does
not. Rejected for being factually wrong about SGLang's coverage.

**Option C — Use Glom for everything, including reasoning.**
Rejected: Glom is a code-execution engine, not an inference runtime. It
cannot serve reasoning models and has no KV-cache or batching. This option
misuses Glom.

**Option C-frontier — Do not serve frontier MoE models on this hardware at
all; route them to a remote API.**
Pros: zero new on-host operational surface; no ~380GB disk commitment; no
NVme tuning. Cons: violates the operator's self-hosted constraint (this
repo exists to run models on the GPU box, not to call out to remote APIs);
adds a network dependency and per-token cost; the agent loop loses the
local, low-latency path it has for every other model class.

**Option C-frontier-2 — Wait for SGLang/vLLM out-of-core MoE support instead
of adopting Colibri.**
Pros: one fewer runtime to operate; stays inside the Docker/SGLang stack.
Cons: no committed timeline for production-grade out-of-core MoE on GB10 in
either runtime; the operator needs frontier MoE now, not on a speculative
upstream roadmap; Colibri is available today and ships an OpenAI-compatible
API so the agent loop is runtime-agnostic.

**Option C-newarch — Wait for SGLang/vLLM to merge support for brand-new
architectures instead of adopting llama.cpp.**
Pros: one fewer runtime to operate; stays inside the server stack. Cons:
no guaranteed timeline for upstream merges; the operator cannot run
brand-new models on day one; GGUF-only quants are never read by the
transformer-based servers, so this option leaves an entire quant family
without a serving path.

**Option D (chosen, amended 2026-08-26) — Tiered runtime selection: SGLang
dense default, FreeToken MoE default (in-VRAM and frontier), vLLM fallback,
llama.cpp bring-up, Colibri emergency fallback, Glom code execution.**
Pros: correct CUDA target for dense default-path models; highest throughput
on the target hardware for supported dense models (SGLang) and MoE models
with agentic workloads (FreeToken); vLLM retained for the models/quant/
parallelism SGLang and FreeToken do not support (especially GPTQ);
llama.cpp covers brand-new architectures and GGUF-only quants on day one;
FreeToken makes frontier MoE servable on this hardware with bandwidth-
adaptive co-execution and semantic-aware caching; Colibri retained as
emergency fallback; Glom owns code execution exclusively. No model is left
without a serving path. Cons: more operational surface (up to five runtimes
plus Glom, net-neutral since FreeToken replaces Colibri as frontier MoE
default); migration cost for the default path; FreeToken GB10/SM103 support
unverified; FreeToken is very new; NVMe disk remains a concern for Colibri
emergency fallback; llama.cpp bring-up models must be migrated to
SGLang/vLLM/FreeToken once supported or they accumulate as debt;
re-validation required across all scenario classes.

**Option E (amendment alternative) — Keep Colibri as frontier MoE default,
do not adopt FreeToken.**
Pros: zero change to the frontier MoE path; avoids early-adopter risk on a
9-day-old engine. Cons: misses semantic-aware caching for agent loops
(purpose-built for this repo's workload); Colibri is a smaller community
project with narrower model support (GLM-5.2 only vs FreeToken's 20+ MoE
models); FreeToken is strictly stronger for the same scenario class.
Rejected — the agentic caching and bandwidth-adaptive execution are
available now and directly address this repo's workload; waiting has an
opportunity cost in throughput.

**Option F (amendment alternative) — FreeToken replaces both Colibri AND
SGLang for all MoE, no SGLang fallback.**
Pros: simplest mental model (one MoE runtime). Cons: removes SGLang as the
documented fallback for in-VRAM MoE; if FreeToken has a critical bug on
GB10, there is no fallback. Rejected — SGLang must be retained as fallback
for in-VRAM MoE and Colibri as emergency fallback for frontier MoE.

## Rollout / Migration

1. **Pin the SGLang image** to `lmsysorg/sglang:v0.5.15-cu130` (or the
   latest validated cu130 tag at rollout time). Do not deploy `latest`.
2. **Refactor the shared library** (`vllm-runner-lib.sh` → multi-runtime
   `runner-lib.sh`) with SGLang, FreeToken, and retained vLLM launch
   functions. Add the SGLang equivalents of `--quantization awq_merlin`,
   `--max-model-len`, and the speculative-decoding flags. Add a
   `run_freetoken()` function for the native Python launch path (no
   Docker). A per-model config encodes which runtime each model uses.
3. **Migrate one dense model script first** — `model-chat.sh`
   (Qwen3-Next-80B, port 8000) — to SGLang as the canary. Smoke-test with
   `test-query.sh` and benchmark throughput at the operator's typical
   concurrency.
4. **If the canary holds**, migrate `model-code.sh`, `model-reasoning.sh`,
   and `model-general.sh` to their default runtimes (SGLang for dense,
   FreeToken for MoE, vLLM for GPTQ) in that order, re-benchmarking each.
   Any model SGLang/FreeToken do not support (or where vLLM is more
   stable) stays on vLLM via the retained launch function — this is the
   documented fallback, not a failure.
5. **Validate FreeToken on the GB10** — install via
   `uv pip install "freetoken[accel]"`, run `ft bench bw` to calibrate
   the bandwidth profile, and serve `nvidia/Qwen3.6-35B-A3B-NVFP4` on
   port 8004. Confirm the server starts and responds. **If SM103 is not
   supported, FreeToken stays proposed-not-accepted and Colibri retains
   the frontier MoE slot.** This is the single biggest validation gate.
6. **Create `model-freetoken-reason.sh`** (Qwen3.5-35B-A3B NVFP4, port
   8004) as the FreeToken canary. Smoke-test with `test-query.sh -p 8004`.
   Benchmark FreeToken vs SGLang for Qwen3.5-35B-A3B at 32-128 concurrent
   agents, 64k-256k context. If FreeToken wins on the agentic workload
   (multi-turn with tool calls), it becomes the MoE default. If SGLang
   wins, FreeToken stays as the frontier MoE runtime only (replacing
   Colibri) and SGLang retains the in-VRAM MoE slot.
7. **Introduce Glom** as the execution target for code-model output
   (regardless of whether the code model is served by SGLang, FreeToken,
   or vLLM). Wire the agent loop to route code output → Glom → result
   back into the next inference call.
8. **Stand up the llama.cpp bring-up path** — build llama.cpp with
   `-DGGML_CUDA=ON`, pin the commit, and create a `model-bringup-*.sh`
   pattern for brand-new architectures / GGUF-only quants on a dedicated
   host port. Document the migration rule: once SGLang/vLLM/FreeToken
   merge support, move the model off llama.cpp.
9. **Reserve a dedicated, fast NVMe path** for the frontier MoE int4 model
   files (~380GB for GLM-5.2). Confirm the path is not contended by other
   workloads; disk speed = token speed (for Colibri emergency fallback).
   FreeToken's hybrid backend uses host RAM + PCIe, so NVMe speed matters
   less but still affects model loading.
10. **Install and pin FreeToken** to a specific PyPI version or git tag.
    Run `ft bench bw` after each version bump to re-calibrate. Create
    `model-freetoken-frontier.sh` (GLM-5.2 NVFP4, port 8005) with
    `--moe-backend hybrid`. Smoke-test via its OpenAI + Anthropic API.
11. **Install and pin Colibri** (emergency fallback) — either the
    prebuilt archive (`colibri-v1.1.0-linux-x86_64.tar.gz`) or a
    from-source build pinned to a recorded commit with `ARCH=native`. Run
    `COLI_MODEL=/nvme/glm52_i4 ./coli doctor` and `./coli plan` to
    confirm RAM/disk/GPU placement. Install `libgomp1` on minimal/cloud
    images so the engine does not exit silently at startup.
12. **Stand up the Colibri emergency fallback endpoint** with
    `COLI_MODEL=/nvme/glm52_i4 ./coli chat --topp 0.85` on a new host
    port. Smoke-test via its OpenAI-compatible API. Only activate if
    FreeToken has a critical bug on GB10.
13. **Update the README model table** (add a "Runtime" column) and the
    operator runbook with a "which runtime does this model use?" decision
    tree, plus per-runtime sections (SGLang dense, FreeToken MoE, vLLM
    fallback, llama.cpp bring-up, Colibri emergency, Glom).
14. **Rollback**: keep the vLLM scripts on a `vllm-` prefix branch until
    all default-path models are validated on SGLang/FreeToken. If a model
    regresses meaningfully on SGLang or FreeToken, that model stays on
    vLLM (the documented fallback) and is tracked as a follow-up rather
    than blocking the others. FreeToken is additive — if the FreeToken
    endpoint regresses or SM103 is not supported, disable the FreeToken
    endpoint and fall back to SGLang (for in-VRAM MoE) or Colibri (for
    frontier MoE) as a temporary measure. Colibri is the emergency
    fallback — if it also fails, route frontier MoE calls to the
    remote-API fallback (Option C-frontier) as a temporary measure.
    llama.cpp bring-up models are inherently temporary — if a bring-up
    model fails, it is not a production regression, just a delayed
    migration.

## To Investigate

- **Exact SGLang flag equivalents** for the vLLM flags currently in use
  (`--quantization awq_merlin`, `--max-model-len 262144`, speculative
  decoding). Confirm against the SGLang server docs and the Weschera guide.
- **Per-model runtime assignment**: for each existing `model-*.sh`, decide
  whether it moves to SGLang (dense default), FreeToken (MoE default), or
  stays on vLLM (fallback). The decision criteria: is the model in the
  runtime's supported matrix? Is the quant supported (FreeToken does not
  support GPTQ)? Is vLLM's stability/parallelism advantage relevant?
- **KV-cache mem-fraction tuning** per model on GB10 128GB — the DGX Spark
  Playbooks note this affects context length and concurrency. FreeToken's
  elastic memory management may reduce the need for manual tuning.
- **Glom installation and isolation model** on the GPU box — does it run
  as a sibling container, a sidecar, or a separate host process? What is
  the network path from the agent loop?
- **NVFP4 vs FP8 vs AWQ-4bit vs GPTQ-int4** per model — SGLang and
  FreeToken both expose NVFP4 on GB10; confirm whether the existing
  AWQ-4bit and GPTQ-int4 weights should be re-quantized to NVFP4 or kept
  as-is. The user's Claude-4.6-Opus reasoning distillation (GPTQ-int4)
  cannot move to FreeToken without re-quantization.
- **Image pinning policy** — decide whether to pin to a specific version
  tag or mirror the image into a private registry. Applies to SGLang,
  vLLM, FreeToken (PyPI version), and llama.cpp (commit hash).
- **FreeToken GB10/SM103 kernel support** — does FreeToken's CUDA kernel
  cache include prebuilt kernels for SM103, or does it JIT-compile? Run
  `ft serve` on the DGX Spark and check for kernel compilation errors.
  This is the single biggest validation gate for FreeToken adoption.
- **FreeToken + Docker** — can FreeToken be containerized for isolation,
  or does it need direct host access to the GPU and host RAM for the
  hybrid backend? If containerizable, add a `run_freetoken_container()`
  function.
- **FreeToken + SGLang coexistence** — can FreeToken and SGLang run
  simultaneously on the GB10 (FreeToken for MoE on port 8004, SGLang for
  dense on port 8000)? VRAM partitioning needs validation.
- **FreeToken semantic caching vs SGLang RadixAttention** — is FreeToken's
  semantic anchor checkpointing a superset of SGLang's RadixAttention, or
  do they address different reuse patterns? If RadixAttention already
  captures the agent-loop prefix reuse, FreeToken's advantage may be
  smaller than expected.
- **FreeToken `ft launch` + existing agent configs** — does `ft launch`
  conflict with existing provider configs for Devin, Cursor, or Continue?
  Can FreeToken serve as the backend for Cursor's local-model autocomplete?
- **NVFP4 re-quantization pipeline** — what tools produce NVFP4
  checkpoints from a BF16 or GPTQ source? NVIDIA's TensorRT Model
  Optimizer? FreeToken's `ft checkpoint`? Can the Jackrong/Codgician
  Claude-4.6-Opus reasoning distillation be converted?
- **llama.cpp CUDA build on GB10** — confirm `-DGGML_CUDA=ON` builds
  cleanly for SM103 and whether the GB10 GPU helps bring-up throughput or
  just contends with the SGLang/FreeToken/vLLM stack. Default assumption:
  run llama.cpp CPU-only for bring-up so the servers keep the full GPU.
- **llama.cpp → server migration trigger**: define the exact condition
  under which a bring-up model on llama.cpp is migrated to
  SGLang/vLLM/FreeToken (e.g., "SGLang/FreeToken merges support + passes
  our smoke test"). Without a trigger, bring-up models accumulate as debt.
- **Colibri GPU mode** — the quickstart says GPU is optional and the
  engine is CPU-default. Investigate whether enabling Colibri's GPU path
  on the GB10 (alongside the SGLang/FreeToken stack) helps frontier MoE
  tokens/sec or just contends with the in-VRAM models for VRAM/compute.
  Default assumption: run Colibri CPU-only so SGLang/FreeToken keep the
  full GPU.
- **Colibri tuning docs** — review `docs/tuning.md` (cache, prefetch,
  speculation) and `docs/ENVIRONMENT.md` (every environment variable) to
  tune the emergency fallback frontier MoE serving path beyond the
  `--topp 0.85` default.
- **NVMe path sizing and contention** — confirm the ~380GB int4 model
  fits with headroom, and that no other workload on the GPU box competes
  for the same NVMe bandwidth during frontier MoE serving (Colibri
  emergency fallback). FreeToken's hybrid backend is less disk-bound.
- **Colibri ARM64 build** — if the GPU box ever moves to an ARM64 host,
  Colibri builds from source unchanged (no x86-only intrinsics), but the
  prebuilt archive is x86_64 only. Record the from-source build as the
  ARM64 path.

## Validation

This decision is the right choice if, after migration:

- Every `model-*.sh` starts and serves on its assigned host port using
  its assigned runtime (SGLang dense default, FreeToken MoE default, or
  vLLM fallback), confirmed by `test-query.sh`.
- Throughput at the operator's typical concurrency (32–128 agents) and
  context length (64k–256k) for SGLang default-path dense models meets or
  exceeds the vLLM baseline on the same hardware, measured by a
  reproducible SGLang-vs-vLLM benchmark (per the Weschera guide).
- Throughput at the operator's typical concurrency for FreeToken
  default-path MoE models meets or exceeds the SGLang baseline for the
  same model, measured by a reproducible FreeToken-vs-SGLang benchmark
  with multi-turn agent workloads (tool calls, thinking blocks). If
  FreeToken does not win on the agentic workload, it stays as the
  frontier MoE runtime only (replacing Colibri) and SGLang retains the
  in-VRAM MoE slot.
- Fallback-path models (vLLM, especially GPTQ) continue to serve at
  parity with their pre-migration baseline — no regression from the
  library refactor.
- Glom successfully executes code produced by the code model (regardless
  of whether SGLang, FreeToken, or vLLM served it) and returns results to
  the agent loop without escaping its sandbox.
- No default-path model regresses by more than 10% on throughput or
  latency versus vLLM; any that does stays on vLLM (the documented
  fallback) and is tracked as a follow-up.
- llama.cpp bring-up path runs a brand-new architecture or GGUF-only
  quant that neither SGLang, vLLM, nor FreeToken supports, via
  `llama-server` on a dedicated host port, with the OpenAI-compatible API
  responding to the agent loop.
- FreeToken serves the frontier MoE model (e.g., GLM-5.2 753B NVFP4) via
  its OpenAI + Anthropic-compatible API on port 8005 with
  `--moe-backend hybrid`, with `ft bench bw` calibrated, and the agent
  loop reaching it as just another endpoint.
- Colibri emergency fallback serves the frontier MoE model (e.g.,
  GLM-5.2 744B int4) via its OpenAI-compatible API on a dedicated host
  port, with `coli doctor` and `coli plan` both passing. Only activated
  if FreeToken has a critical bug on GB10.
- Frontier MoE tokens/sec on FreeToken is bounded by the calibrated
  bandwidth profile (not by a software bottleneck), and the `hybrid`
  backend's CPU-GPU split is confirmed optimal by `ft bench bw`.
- FreeToken's semantic-aware caching measurably reduces KV recomputation
  for multi-turn agent loops with tool calls, compared to SGLang's
  RadixAttention on the same workload. If the advantage is not
  measurable, FreeToken's in-VRAM MoE default is reconsidered at the
  review date.

## Review Schedule

- **6 months after acceptance** (target: 2027-02-02) — re-benchmark against
  the latest vLLM cu130 release, the latest SGLang cu130 release, and the
  latest FreeToken release. If vLLM closes the throughput gap on
  Blackwell for dense models, reconsider the SGLang dense default. If
  SGLang adds semantic-aware agentic caching by then, reconsider the
  FreeToken MoE default. Separately, re-evaluate Colibri against any
  out-of-core MoE support that has shipped in SGLang/vLLM/FreeToken by
  then; if any runtime can now serve the frontier MoE model at parity
  with FreeToken's hybrid backend, consider consolidating back to fewer
  runtimes. Re-evaluate llama.cpp bring-up models: any that
  SGLang/vLLM/FreeToken now support should have been migrated.
- **Trigger review early** if: NVIDIA ships a new GB10/GB20 container that
  changes the cu130 requirement; SGLang drops cu130 support; FreeToken
  drops GB10/SM103 support or becomes deprecated; Glom becomes a
  deprecated/unsupported project; Colibri becomes deprecated/unsupported
  (removing the emergency fallback); llama.cpp becomes deprecated/
  unsupported; or a frontier MoE model larger than GLM-5.2 becomes the
  operator's baseline and the ~380GB NVMe footprint is no longer
  sufficient.

## Notes

- Current state of decisions belongs in `decisions.md` (if/when created).
- Implementation details belong in `internal-docs/decision-records/gdr/*.md`
  and, once migration begins, in the refactored multi-runtime
  `runner-lib.sh`, per-model scripts, the FreeToken launch/runbook
  section, the llama.cpp bring-up pattern, and the Colibri emergency
  fallback launch/runbook section.
- This ADR is **proposed**, not accepted. Acceptance requires: the canary
  migration of `model-chat.sh` to SGLang to pass its smoke test and
  throughput benchmark; at least one fallback model confirmed still
  serving on vLLM via the refactored library; FreeToken confirmed
  installing and serving on the GB10 (SM103 validation gate); the
  FreeToken canary (`model-freetoken-reason.sh`) passing its smoke test
  and agentic-workload benchmark; the llama.cpp bring-up path confirmed
  running a model no server supports; and the Colibri emergency fallback
  endpoint to pass `coli doctor`, `coli plan`, and an OpenAI-compatible
  API smoke test.

## References

- NVIDIA — "SGLang for Inference | DGX Spark": https://build.nvidia.com/spark/sglang
- DGX Spark Playbooks — SGLang: https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/sglang/README.md
- Weschera — Qwen3.6 + SGLang on DGX Spark: https://github.com/Weschera/qwen-sglang-dgx-spark
- FreeToken — paper: https://arxiv.org/abs/2608.16157
- FreeToken — repository (edge-native MoE serving engine): https://github.com/FlashML-org/FreeToken
- FreeToken — website: https://www.flashml.ai/
- FreeToken — Quickstart: https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md
- FreeToken — Supported models: https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md
- FreeToken — CLI reference: https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md
- FreeToken — Install guide: https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md
- Amendment ADR: [`adr-20260826-freetoken-runtime-amendment.md`](adr-20260826-freetoken-runtime-amendment.md)
- Research note: [`note-freetoken-edge-native-moe-serving.md`](../../../note-freetoken-edge-native-moe-serving.md)
- Colibri — Quick Start (emergency fallback frontier MoE disk-streaming engine): https://github.com/JustVugg/colibri/blob/main/docs/quickstart.md
- Colibri — repository: https://github.com/JustVugg/colibri
- Colibri — Releases (prebuilt archives): https://github.com/JustVugg/colibri/releases
- llama.cpp — repository (ggml/GGUF first-mover runtime): https://github.com/ggerganov/llama.cpp
- vLLM — repository (fallback inference runtime): https://github.com/vllm-project/vllm
- LMSYS — Optimizing GPT-OSS on DGX Spark (video): https://www.youtube.com/watch?v=ApIVoTuWIss
- LMSYS — SGLang Cookbook (video): https://www.youtube.com/watch?v=R74kx7GKEbs
- Michael Nygard — Documenting Architecture Decisions: https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
- This repo's current vLLM setup: [`README.md`](../../../README.md), [`AGENTS.md`](../../../AGENTS.md)

<!-- vim: set ft=markdown: -->
