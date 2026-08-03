---
modeline: "vim: set ft=markdown:"
title: "GDR: Tiered LLM Runtime Selection on DGX Spark — SGLang, vLLM, llama.cpp, Colibri, Glom"
gdr-id: "gdr20260802001"
slug: "tiered-llm-runtime-selection"
url: "https://github.com/levonk/levonk-ai-playground/blob/main/internal-docs/adr/2026/08/adr-202608021744-sglang-glom-runtime-mapping.md"
synopsis: "Select the LLM runtime per scenario on DGX Spark: SGLang (cu130) as the default for supported in-VRAM models needing max throughput; vLLM as the fallback for models/quants/parallelism SGLang does not support or where production maturity matters; llama.cpp (ggml/GGUF) for brand-new architectures or GGUF-only quants not yet in SGLang/vLLM; Colibri for frontier MoE models that must stream experts from disk and cannot fit in 128GB VRAM; Glom only as a sandboxed code-execution engine for code-model outputs."
author: "https://github.com/levonk"
date-created: "2026-08-02"
date-updated: "2026-08-02"
date-review: "2027-02-02"
date-triggers: ["2026-11-02"]
version: "0.1.0"
status: "proposed"
aliases: []
tags: [doc/architecture/gdr]
supersedes: []
superseded-by: []
related-to: []
scope:
  impact-scope:
    - "model-*.sh runner scripts"
    - "vllm-runner-lib.sh shared library"
    - "Docker base image (nvcr.io/nvidia/vllm → lmsysorg/sglang:latest-cu130, with vLLM retained as fallback)"
    - "Host port allocation per model"
    - "Multi-agent orchestration layer"
    - "Frontier MoE model serving (Colibri engine + disk-streaming path)"
    - "llama.cpp (ggml/GGUF) bring-up path for brand-new architectures"
  excluded-scope:
    - "Model weights and quantization format selection"
    - "Agent framework choice (Hermes/Paperclip/etc.)"
    - "Host OS and NVIDIA driver configuration"
hardware:
  target: "NVIDIA DGX Spark (Blackwell GB10 128GB / GB20)"
  container: "lmsysorg/sglang:latest-cu130 (SM103 / cu130 required) — default; vLLM retained as fallback"
  validated-precisions: ["NVFP4", "FP8", "AWQ-4bit", "GPTQ", "GGUF"]
  frontier-engine: "Colibri (JustVugg/colibri) — C engine, CPU-default, GPU-optional, streams MoE experts from NVMe"
  new-arch-engine: "llama.cpp (ggml/GGUF) — first-mover support for brand-new architectures and GGUF-only quants"
---

# Decision Record: Tiered LLM Runtime Selection on DGX Spark — SGLang, vLLM, llama.cpp, Colibri, Glom

**Filename:** `adr-202608021744-sglang-glom-runtime-mapping.md`

- belongs in `internal-docs/adr/2026/08/`

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
served by either vLLM or SGLang on this hardware; and (4) what to do for
**brand-new model architectures** or **GGUF-only quantizations** that
neither SGLang nor vLLM support yet.

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
  answers or precision — it is still the full model.
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
  without an out-of-core path. The runtime for these models must stream
  experts from NVMe on demand.
- **Brand-new architectures and GGUF-only quants exist.** When a model
  architecture or quantization format is too new for SGLang and vLLM, the
  operator still needs to run it. The runtime selection must include a
  first-mover path (llama.cpp/ggml) for these cases rather than blocking on
  upstream support.
- **No cross-ecosystem runtime mixing**: this repo maps within-ecosystem
  alternatives (e.g., pip→uv, npm→pnpm) but never across ecosystems. The
  same principle applies here — pick a default inference runtime per
  scenario class and document the fallbacks, rather than arbitrarily
  splitting models across runtimes. SGLang, vLLM, llama.cpp, and Colibri
  each own a distinct scenario class, not a competing slice of the same
  class.

## Decision

Select the LLM runtime **per scenario**, not per model type alone. SGLang
is the **default** for in-VRAM inference where supported; vLLM, llama.cpp,
and Colibri are the documented fallbacks for the scenarios SGLang does not
cover. Glom is the only runtime for code execution.

The scenario-to-runtime mapping is:

| Scenario                                                      | Default runtime | Fallback / alternative                  |
|---------------------------------------------------------------|:---------------:|:---------------------------------------:|
| Supported model, max throughput on Blackwell, multi-agent long-context | **SGLang (cu130)** | vLLM (if maturity/stability preferred) |
| Model/quant/parallelism SGLang does not support (GPTQ, compressed-tensors, multi-GPU pipeline parallel, some LoRA serving) | **vLLM** | SGLang (once support lands) |
| Brand-new architecture not yet in SGLang or vLLM              | **llama.cpp (ggml)** | SGLang/vLLM (once upstream merges)     |
| GGUF-only quant (Q4_K_M, Q8_0, IQ4_XS, etc.)                  | **llama.cpp (ggml)** | Re-quantize to a server-supported format |
| Frontier MoE that does not fit in 128GB VRAM (e.g. GLM-5.2 744B) | **Colibri** | Remote API (temporary; violates self-hosted) |
| Code execution (run code produced by a code model)            | **Glom** | — (no alternative; Glom is the only execution sandbox) |

Per model type, the defaults are:

| Model Type                  | SGLang | vLLM (fallback) | llama.cpp (new-arch/GGUF) | Colibri (frontier MoE) | Glom (execute) |
|-----------------------------|:------:|:---------------:|:-------------------------:|:----------------------:|:--------------:|
| General                     | default | fallback        | if too new                | no                     | no             |
| Chat                        | default | fallback        | if too new                | no                     | no             |
| Reasoning                   | default | fallback        | if too new                | no                     | no             |
| Code                        | default (generate) | fallback (generate) | if too new (generate) | no          | yes (execute)  |
| Frontier MoE (e.g. GLM-5.2) | no     | no              | no                        | default                | no             |

In short: **SGLang is the default for what fits in VRAM and is supported;
vLLM is the fallback when SGLang cannot serve it; llama.cpp is the
first-mover for what is too new for either; Colibri serves what is too big
to fit in VRAM at all; Glom runs the code a code model produces.** No
runtime is asked to do another's job, and no model is left without a
serving path.

## Rationale

**Why SGLang is the default for in-VRAM inference on DGX Spark (not the
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
  SGLang catches up. When a model is not in SGLang's supported matrix,
  vLLM is the answer, not a workaround.
- **Quantization format coverage**: GPTQ, some AWQ variants,
  compressed-tensors, and FP8 layouts where vLLM's kernels are more
  mature. SGLang's NVFP4 path on GB10 is a real advantage, but it is not
  universal across every quant.
- **Multi-GPU pipeline parallelism and some LoRA-serving features** where
  vLLM is ahead.
- **Production maturity**: when the operator needs the known-good baseline,
  not the throughput frontier, vLLM is the right choice. The decision is
  "SGLang by default, vLLM when SGLang cannot serve it or stability wins,"
  not "SGLang only."

**Why llama.cpp (ggml/GGUF) is the first-mover path for brand-new
architectures and GGUF-only quants:**

- **First-mover architecture support**: the ggml codebase is small and the
  community implements new model architectures fast, often before SGLang or
  vLLM merge support. If a model dropped last week, llama.cpp may be the
  only thing that runs it.
- **GGUF-only quantizations**: GGUF quants (Q4_K_M, Q8_0, IQ4_XS, etc.)
  are not read by the transformer-based servers at all. llama.cpp is the
  only runtime that reads them.
- **CPU-first / single-file portable**: GGUF is one file; useful for quick
  bring-up before committing to a server runtime. llamafile, ollama, and
  koboldcpp are the same ggml family under different launchers.
- **Not a replacement for SGLang/vLLM on supported models**: llama.cpp
  lacks the batching, KV-cache reuse, and multi-agent concurrency of the
  server runtimes. It is the bring-up/new-arch path, not the production
  serving path for models the servers already support.

**Why Glom is scoped to execution only:**

- Glom is a code-execution engine, not an inference runtime. Asking it to
  serve general/chat/reasoning models would misuse it and lose batching,
  KV-cache reuse, and concurrency.
- The only place Glom adds value is **executing** the code that a
  code model produces — sandboxed Python/JS/shell evaluation and tool
  calls, regardless of whether the code model was served by SGLang, vLLM,
  or llama.cpp.

**Why Colibri is scoped to frontier MoE only:**

- A 744B-parameter MoE model at int4 is ~380GB on disk and physically
  cannot be resident in the GB10's 128GB VRAM. Neither vLLM nor SGLang can
  serve it on this hardware without an out-of-core path; Colibri's
  disk-streaming design is purpose-built for exactly this case.
- Colibri streams the model's experts from NVMe on demand, so streaming
  speed = token speed. A fast NVMe SSD is the single biggest factor in
  tokens/sec, and placement only changes speed — never the model's answers
  or precision. It is still the full model.
- It is a single C program with OpenMP and no x86-only intrinsics (builds
  unchanged on ARM64/Graviton), ships an OpenAI-compatible API + web
  dashboard, and is driven by a `coli` launcher. This makes it a clean,
  low-dependency addition to the host alongside the SGLang/vLLM Docker
  stack.
- It is **not** a replacement for SGLang/vLLM on in-VRAM models. For models
  that fit in 128GB VRAM, the server runtimes' batching, KV-cache reuse,
  speculative decoding, and GB10-native cu130 container beat a CPU-default,
  disk-streamed engine on throughput and latency. Colibri is only the
  answer for the size class the servers cannot address at all.

**Why a tiered default-with-fallbacks model instead of one runtime for
everything:**

- No single runtime covers every scenario. SGLang wins on throughput for
  supported models on Blackwell but does not support every model/quant;
  vLLM has broader support but is not cu130-native and is slower on this
  hardware; llama.cpp is the only path for brand-new architectures and
  GGUF quants but lacks server-grade concurrency; Colibri is the only path
  for frontier MoE but is disk-bound. Pretending any one of them is the
  only runtime leaves real models without a serving path.
- The tiered model keeps one **default** per scenario class (so the
  operator is not making a fresh decision every time) while documenting
  the **fallback** (so the operator is never blocked).

**Trade-offs and risks:**

- **Migration cost**: the default path moves `model-*.sh` scripts and the
  shared library from vLLM to SGLang. The shared library's vLLM-specific
  flags (`--quantization awq_merlin`, `--max-model-len`, speculative
  decoding flags) need SGLang equivalents. vLLM scripts are retained (not
  deleted) as the fallback path.
- **Container image change (default path)**: `nvcr.io/nvidia/vllm:26.04-py3`
  → `lmsysorg/sglang:latest-cu130`. The `latest` tag is a floating
  reference; pinning to a specific version (e.g., `v0.5.15-cu130`) is
  required for reproducibility. vLLM's image is retained for fallback
  models.
- **More operational surface**: the operator now manages up to four
  runtimes (SGLang, vLLM fallback, llama.cpp bring-up, Colibri frontier
  MoE) plus Glom. Each has its own pin discipline and health checks. This
  is the honest cost of covering every scenario instead of pretending one
  runtime suffices.
- **Glom is a new dependency**: it must be installed, networked to the
  agent loop, and kept isolated from the inference runtimes.

## Technical Approach

### Container and launch (default path: SGLang)

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

Models that SGLang does not support (or where vLLM's stability/quant
coverage is preferred) keep their existing `model-*.sh` launching
`nvcr.io/nvidia/vllm:26.04-py3` (or a cu130-native vLLM image once one
ships). The shared library retains a vLLM launch function alongside the
SGLang one. The decision of which runtime a given model uses is encoded in
the model script (or a per-model config), not made at runtime.

### Bring-up path: llama.cpp (ggml/GGUF)

For brand-new architectures or GGUF-only quants not yet in SGLang/vLLM,
use llama.cpp (or a ggml-family launcher: llamafile, ollama, koboldcpp)
as the bring-up runtime:

```bash
# Build llama.cpp with CUDA support for the GB10
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release

# Serve a GGUF model with the OpenAI-compatible server
./build/bin/llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 80XX
```

This is the first-mover path: once SGLang or vLLM merges support for the
architecture/quant, migrate the model off llama.cpp to the server runtime
for batching and multi-agent concurrency. llama.cpp is not the production
serving path for models the servers already support.

### Model-to-runtime routing in the agent loop

- "Write code" → code model on its host port (SGLang default; vLLM/llama.cpp fallback)
- "Run code"   → Glom (sandboxed execution of code-model output, regardless of which runtime served it)
- "Think deeply" → reasoning model (SGLang default; vLLM/llama.cpp fallback)
- "Chat" / "General tasks" → general/chat model (SGLang default; vLLM/llama.cpp fallback)
- "Frontier MoE / huge model" → Colibri (disk-streamed, OpenAI-compatible API)
- "Brand-new architecture / GGUF-only quant" → llama.cpp (bring-up; migrate to SGLang/vLLM once supported)

The agent framework (Hermes/Paperclip/etc.) routes outputs from the code
model into Glom for execution, and feeds execution results back into the
next inference call. Frontier MoE calls go to Colibri's OpenAI-compatible
endpoint; bring-up calls go to llama.cpp's OpenAI-compatible server. From
the agent's perspective, every runtime exposes the same OpenAI API shape
on a different host port, so routing is URL config, not framework changes.

### Frontier MoE serving with Colibri

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
(or a cu130-native vLLM image once one ships) for models that stay on vLLM.

Pin llama.cpp to a recorded commit hash when building from source
(`-DGGML_CUDA=ON` for the GB10), or to a specific release tag. Bring-up
models on llama.cpp should be migrated to SGLang/vLLM once upstream support
lands, at which point the llama.cpp pin for that model is retired.

Pin Colibri to a specific release archive (e.g.,
`colibri-v1.1.0-linux-x86_64.tar.gz`) rather than building from `main`,
and record the commit hash if building from source with `ARCH=native`.
Colibri's `coli doctor` should pass before the frontier MoE endpoint is
advertised to the agent loop.

## Affected Components

- **`vllm-runner-lib.sh`** — refactored into a multi-runtime shared library
  (`runner-lib.sh` or `sglang-runner-lib.sh` with a retained vLLM launch
  function); vLLM-specific flags kept for fallback models, SGLang flags
  added for default-path models.
- **`model-chat.sh`, `model-code.sh`, `model-reasoning.sh`, `model-general.sh`**
  — each migrated to its default runtime (SGLang where supported, vLLM
  where not) on its existing host port (8000–8003). A per-model config
  encodes which runtime each model uses.
- **Docker base images** — `nvcr.io/nvidia/vllm:26.04-py3` retained for
  fallback models; `lmsysorg/sglang:<pinned>-cu130` added for default-path
  models.
- **llama.cpp bring-up path** — a new `model-bringup-*.sh` pattern (or a
  `llama-runner-lib.sh` function) for brand-new architectures / GGUF-only
  quants, using `llama-server` on a dedicated host port.
- **README model table** — gains a "Runtime" column (SGLang / vLLM /
  llama.cpp / Colibri) alongside the existing "Special Config" column;
  the "Special Config" column moves from vLLM-only flags to per-runtime
  flags; new rows for frontier MoE (Colibri) and bring-up (llama.cpp).
- **`.env` / launch env vars** — `HOST_PORT`, `MODEL`, `ACCT`,
  `GPU_MEMORY_UTILIZATION` semantics may shift per runtime; Colibri adds
  `COLI_MODEL` (path to the on-disk int4 model) and `--topp 0.85` as a
  default for disk-bound frontier serving; llama.cpp adds `GGUF_PATH`
  and `llama-server` flags.
- **Agent loop / orchestrator** — gains a Glom execution step for code
  model outputs, a Colibri endpoint config for frontier MoE calls, and a
  llama.cpp endpoint config for bring-up calls (all OpenAI-compatible,
  so URL-only).
- **NVMe storage layout** — a dedicated, fast NVMe path must be reserved
  for the ~380GB int4 frontier MoE model files; disk speed = token speed.
- **Operator runbook** — startup, smoke-test, and troubleshooting steps
  per runtime (SGLang, vLLM fallback, llama.cpp bring-up, Colibri); a
  decision tree for "which runtime does this model use?"

## Consequences

### Negative

- **Migration cost for the default path**: the `model-*.sh` scripts moving
  to SGLang and the shared library must be rewritten for SGLang's launch
  flags. No incremental vLLM→SGLang shim; the launch flags and server API
  differ enough that a clean rewrite is cheaper than a compatibility layer.
  vLLM scripts are retained (not deleted) for fallback models, so the
  library carries both launch functions.
- **More operational surface**: the operator now manages up to four
  runtimes (SGLang default, vLLM fallback, llama.cpp bring-up, Colibri
  frontier MoE) plus Glom. Each has its own pin discipline and health
  checks. This is the honest cost of covering every scenario instead of
  pretending one runtime suffices.
- **Colibri is a native C engine on the host (outside Docker)**, with its
  own build/pin discipline, `libgomp1` runtime dep, and
  `coli doctor`/`coli plan` health checks.
- **llama.cpp bring-up models must be migrated** to SGLang/vLLM once
  upstream support lands, or they accumulate as technical debt on a
  non-production runtime.
- **NVMe disk is the frontier-MoE bottleneck.** Tokens/sec is bounded by
  streaming speed; on a slow or shared disk, generation can be well under
  1 token/sec. The GPU box must reserve a dedicated, fast NVMe path for
  the ~380GB int4 model files, and disk contention with other workloads
  directly degrades frontier MoE latency.
- **Re-validation of every model**: each `model-*.sh` moving to SGLang
  must be re-benchmarked to confirm the throughput and context-length
  claims hold for the specific weights this repo uses; fallback models
  stay on vLLM and are re-validated against the latest vLLM release;
  frontier MoE must be benchmarked on Colibri separately, with `coli plan`
  confirming the RAM/disk/GPU placement; bring-up models on llama.cpp are
  validated only enough to confirm correctness, not throughput.
- **Image and engine pinning discipline required**: `latest-cu130` is a
  floating tag, Colibri's `main` branch moves, and llama.cpp commits move
  fast. Without pinning, a bad upstream push can break the GPU box on next
  pull/build.

### Positive

- **Higher throughput on the same hardware for supported models**: SGLang's
  scheduler, KV-cache reuse, and speculative decoding are tuned for
  Blackwell and the multi-agent, long-context workloads this repo targets.
- **Correct CUDA target by default**: cu130 / SM103 is the right container
  for GB10/GB20, removing the manual cu130 workaround vLLM requires for
  default-path models.
- **No model is left without a serving path.** SGLang covers supported
  in-VRAM models; vLLM covers the models/quant/parallelism SGLang does
  not; llama.cpp covers brand-new architectures and GGUF-only quants;
  Colibri covers frontier MoE that does not fit in VRAM. The tiered model
  is honest about the fact that no single runtime covers everything.
- **Clean separation of concerns**: SGLang/vLLM/llama.cpp own inference
  (each for their scenario); Glom owns code execution; Colibri owns
  disk-streamed frontier MoE. Non-overlapping scenario classes.
- **NVFP4 support**: SGLang v0.5.15-cu130 adds GB10-native Qwen NVFP4,
  opening a precision path vLLM does not expose on this hardware.
- **Frontier MoE is now servable on this hardware at all.** Without
  Colibri, a 744B MoE model is simply out of reach on a 128GB-VRAM box.
  Colibri's disk-streaming makes it possible, and its OpenAI-compatible
  API means the agent loop treats it as just another endpoint.
- **Brand-new architectures are servable on day one** via llama.cpp,
  instead of blocking on SGLang/vLLM upstream merges.

### Neutral

- **Host port allocation is unchanged**: each in-VRAM model keeps its
  unique port (8000–8003), so the multi-model-simultaneous-run property
  is preserved. Colibri and llama.cpp bring-up each get new host ports
  for their OpenAI-compatible endpoints, extending the same pattern.
- **`.env` and `HUGGING_FACE_HUB_TOKEN` handling is unchanged**: SGLang
  and vLLM read the same HF token for private model pulls. Colibri reads
  model files from `COLI_MODEL` on disk, so it does not need the HF token
  at serve time (only at model-prep time, which is a one-time Python
  step). llama.cpp reads GGUF files from disk, same as Colibri.

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

**Option D (chosen) — Tiered runtime selection: SGLang default, vLLM
fallback, llama.cpp bring-up, Colibri frontier MoE, Glom code execution.**
Pros: correct CUDA target for default-path models; highest throughput on
the target hardware for supported models; vLLM retained for the
models/quant/parallelism SGLang does not support; llama.cpp covers
brand-new architectures and GGUF-only quants on day one; Colibri makes
frontier MoE servable on this hardware at all; Glom owns code execution
exclusively. No model is left without a serving path. Cons: more
operational surface (up to four runtimes plus Glom); migration cost for
the default path; NVMe disk becomes a first-class operational concern for
frontier MoE; llama.cpp bring-up models must be migrated to SGLang/vLLM
once supported or they accumulate as debt; re-validation required across
all scenario classes.

## Rollout / Migration

1. **Pin the SGLang image** to `lmsysorg/sglang:v0.5.15-cu130` (or the
   latest validated cu130 tag at rollout time). Do not deploy `latest`.
2. **Refactor the shared library** (`vllm-runner-lib.sh` → multi-runtime
   `runner-lib.sh`) with both a SGLang launch function and the retained
   vLLM launch function. Add the SGLang equivalents of
   `--quantization awq_merlin`, `--max-model-len`, and the
   speculative-decoding flags. A per-model config encodes which runtime
   each model uses.
3. **Migrate one model script first** — `model-chat.sh` (Qwen3-Next-80B,
   port 8000) — to SGLang as the canary. Smoke-test with `test-query.sh`
   and benchmark throughput at the operator's typical concurrency.
4. **If the canary holds**, migrate `model-code.sh`, `model-reasoning.sh`,
   and `model-general.sh` to SGLang in that order, re-benchmarking each.
   Any model SGLang does not support (or where vLLM is more stable) stays
   on vLLM via the retained launch function — this is the documented
   fallback, not a failure.
5. **Introduce Glom** as the execution target for code-model output
   (regardless of whether the code model is served by SGLang or vLLM).
   Wire the agent loop to route code output → Glom → result back into the
   next inference call.
6. **Stand up the llama.cpp bring-up path** — build llama.cpp with
   `-DGGML_CUDA=ON`, pin the commit, and create a `model-bringup-*.sh`
   pattern for brand-new architectures / GGUF-only quants on a dedicated
   host port. Document the migration rule: once SGLang/vLLM merge support,
   move the model off llama.cpp.
7. **Reserve a dedicated, fast NVMe path** for the frontier MoE int4 model
   files (~380GB for GLM-5.2). Confirm the path is not contended by other
   workloads; disk speed = token speed.
8. **Install and pin Colibri** — either the prebuilt archive
   (`colibri-v1.1.0-linux-x86_64.tar.gz`) or a from-source build pinned to
   a recorded commit with `ARCH=native`. Run `COLI_MODEL=/nvme/glm52_i4
   ./coli doctor` and `./coli plan` to confirm RAM/disk/GPU placement
   before serving. Install `libgomp1` on minimal/cloud images so the
   engine does not exit silently at startup.
9. **Stand up the frontier MoE endpoint** with
   `COLI_MODEL=/nvme/glm52_i4 ./coli chat --topp 0.85` on a new host port.
   Smoke-test via its OpenAI-compatible API; confirm the agent loop can
   reach it as just another OpenAI endpoint.
10. **Update the README model table** (add a "Runtime" column) and the
    operator runbook with a "which runtime does this model use?" decision
    tree, plus per-runtime sections (SGLang, vLLM fallback, llama.cpp
    bring-up, Colibri, Glom).
11. **Rollback**: keep the vLLM scripts on a `vllm-` prefix branch until
    all default-path models are validated on SGLang. If a model regresses
    meaningfully on SGLang, that model stays on vLLM (the documented
    fallback) and is tracked as a follow-up rather than blocking the
    others. Colibri is additive — if the frontier MoE endpoint regresses
    or the NVMe path proves too slow, disable the Colibri endpoint and
    route frontier MoE calls to the remote-API fallback (Option
    C-frontier) as a temporary measure. llama.cpp bring-up models are
    inherently temporary — if a bring-up model fails, it is not a
    production regression, just a delayed migration.

## To Investigate

- **Exact SGLang flag equivalents** for the vLLM flags currently in use
  (`--quantization awq_merlin`, `--max-model-len 262144`, speculative
  decoding). Confirm against the SGLang server docs and the Weschera guide.
- **Per-model runtime assignment**: for each existing `model-*.sh`, decide
  whether it moves to SGLang (default) or stays on vLLM (fallback). The
  decision criteria: is the model in SGLang's supported matrix? Is the
  quant supported? Is vLLM's stability/parallelism advantage relevant?
- **KV-cache mem-fraction tuning** per model on GB10 128GB — the DGX Spark
  Playbooks note this affects context length and concurrency.
- **Glom installation and isolation model** on the GPU box — does it run
  as a sibling container, a sidecar, or a separate host process? What is
  the network path from the agent loop?
- **NVFP4 vs FP8 vs AWQ-4bit** per model — SGLang exposes NVFP4 on GB10;
  confirm whether the existing AWQ-4bit weights should be re-quantized to
  NVFP4 or kept as-is.
- **Image pinning policy** — decide whether to pin to a specific version
  tag or mirror the image into a private registry. Applies to SGLang,
  vLLM, and llama.cpp (commit hash).
- **llama.cpp CUDA build on GB10** — confirm `-DGGML_CUDA=ON` builds
  cleanly for SM103 and whether the GB10 GPU helps bring-up throughput or
  just contends with the SGLang/vLLM Docker stack. Default assumption: run
  llama.cpp CPU-only for bring-up so the servers keep the full GPU.
- **llama.cpp → server migration trigger**: define the exact condition
  under which a bring-up model on llama.cpp is migrated to SGLang/vLLM
  (e.g., "SGLang merges support + passes our smoke test"). Without a
  trigger, bring-up models accumulate as debt.
- **Colibri GPU mode** — the quickstart says GPU is optional and the
  engine is CPU-default. Investigate whether enabling Colibri's GPU path
  on the GB10 (alongside the SGLang Docker stack) helps frontier MoE
  tokens/sec or just contends with the in-VRAM models for VRAM/compute.
  Default assumption: run Colibri CPU-only so SGLang keeps the full GPU.
- **Colibri tuning docs** — review `docs/tuning.md` (cache, prefetch,
  speculation) and `docs/ENVIRONMENT.md` (every environment variable) to
  tune the frontier MoE serving path beyond the `--topp 0.85` default.
- **NVMe path sizing and contention** — confirm the ~380GB int4 model
  fits with headroom, and that no other workload on the GPU box competes
  for the same NVMe bandwidth during frontier MoE serving.
- **Colibri ARM64 build** — if the GPU box ever moves to an ARM64 host,
  Colibri builds from source unchanged (no x86-only intrinsics), but the
  prebuilt archive is x86_64 only. Record the from-source build as the
  ARM64 path.

## Validation

This decision is the right choice if, after migration:

- Every `model-*.sh` starts and serves on its assigned host port using
  its assigned runtime (SGLang default or vLLM fallback), confirmed by
  `test-query.sh`.
- Throughput at the operator's typical concurrency (32–128 agents) and
  context length (64k–256k) for default-path models meets or exceeds the
  vLLM baseline on the same hardware, measured by a reproducible
  SGLang-vs-vLLM benchmark (per the Weschera guide).
- Fallback-path models (vLLM) continue to serve at parity with their
  pre-migration baseline — no regression from the library refactor.
- Glom successfully executes code produced by the code model (regardless
  of whether SGLang or vLLM served it) and returns results to the agent
  loop without escaping its sandbox.
- No default-path model regresses by more than 10% on throughput or
  latency versus vLLM; any that does stays on vLLM (the documented
  fallback) and is tracked as a follow-up.
- llama.cpp bring-up path runs a brand-new architecture or GGUF-only
  quant that neither SGLang nor vLLM supports, via `llama-server` on a
  dedicated host port, with the OpenAI-compatible API responding to the
  agent loop.
- Colibri serves the frontier MoE model (e.g., GLM-5.2 744B int4) via its
  OpenAI-compatible API on a dedicated host port, with `coli doctor` and
  `coli plan` both passing, and the agent loop reaching it as just
  another OpenAI endpoint.
- Frontier MoE tokens/sec is bounded by NVMe streaming speed (not by a
  software bottleneck), and `--topp 0.85` measurably raises tokens/sec
  with no quality regression versus default sampling.

## Review Schedule

- **6 months after acceptance** (target: 2027-02-02) — re-benchmark against
  the latest vLLM cu130 release and the latest SGLang cu130 release. If
  vLLM closes the throughput gap on Blackwell, reconsider the default
  in-VRAM runtime choice. Separately, re-evaluate Colibri against any
  out-of-core MoE support that has shipped in SGLang/vLLM by then; if
  either runtime can now serve the frontier MoE model at parity, consider
  consolidating back to fewer runtimes. Re-evaluate llama.cpp bring-up
  models: any that SGLang/vLLM now support should have been migrated.
- **Trigger review early** if: NVIDIA ships a new GB10/GB20 container that
  changes the cu130 requirement; SGLang drops cu130 support; Glom
  becomes a deprecated/unsupported project; Colibri becomes
  deprecated/unsupported; llama.cpp becomes deprecated/unsupported; or a
  frontier MoE model larger than GLM-5.2 becomes the operator's baseline
  and the ~380GB NVMe footprint is no longer sufficient.

## Notes

- Current state of decisions belongs in `decisions.md` (if/when created).
- Implementation details belong in `internal-docs/decision-records/gdr/*.md`
  and, once migration begins, in the refactored multi-runtime
  `runner-lib.sh`, per-model scripts, the llama.cpp bring-up pattern, and
  the Colibri launch/runbook section.
- This ADR is **proposed**, not accepted. Acceptance requires: the canary
  migration of `model-chat.sh` to SGLang to pass its smoke test and
  throughput benchmark; at least one fallback model confirmed still
  serving on vLLM via the refactored library; the llama.cpp bring-up path
  confirmed running a model neither server supports; and the Colibri
  frontier MoE endpoint to pass `coli doctor`, `coli plan`, and an
  OpenAI-compatible API smoke test.

## References

- NVIDIA — "SGLang for Inference | DGX Spark": https://build.nvidia.com/spark/sglang
- DGX Spark Playbooks — SGLang: https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/sglang/README.md
- Weschera — Qwen3.6 + SGLang on DGX Spark: https://github.com/Weschera/qwen-sglang-dgx-spark
- Colibri — Quick Start (frontier MoE disk-streaming engine): https://github.com/JustVugg/colibri/blob/main/docs/quickstart.md
- Colibri — repository: https://github.com/JustVugg/colibri
- Colibri — Releases (prebuilt archives): https://github.com/JustVugg/colibri/releases
- llama.cpp — repository (ggml/GGUF first-mover runtime): https://github.com/ggerganov/llama.cpp
- vLLM — repository (fallback inference runtime): https://github.com/vllm-project/vllm
- LMSYS — Optimizing GPT-OSS on DGX Spark (video): https://www.youtube.com/watch?v=ApIVoTuWIss
- LMSYS — SGLang Cookbook (video): https://www.youtube.com/watch?v=R74kx7GKEbs
- Michael Nygard — Documenting Architecture Decisions: https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
- This repo's current vLLM setup: [`README.md`](../../../README.md), [`AGENTS.md`](../../../AGENTS.md)

<!-- vim: set ft=markdown: -->
