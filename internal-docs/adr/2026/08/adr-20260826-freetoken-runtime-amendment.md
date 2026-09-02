---
modeline: "vim: set ft=markdown:"
title: "ADR Amendment: Add FreeToken as MoE Serving Runtime, Replace Colibri"
gdr-id: "gdr20260826001"
slug: "freetoken-runtime-amendment"
url: "https://github.com/levonk/levonk-ai-playground/blob/main/internal-docs/adr/2026/08/adr-20260826-freetoken-runtime-amendment.md"
synopsis: "Amends ADR-202608021744 to add FreeToken (FlashML) as the default runtime for MoE serving — both in-VRAM and frontier — replacing Colibri. SGLang is retained for dense models and non-MoE architectures. vLLM is retained for GPTQ and quantization formats FreeToken does not support. FreeToken's semantic-aware agentic caching and bandwidth-adaptive CPU-GPU co-execution make it the strongest runtime for the MoE + multi-agent workload class this repo targets."
author: "https://github.com/levonk"
date-created: "2026-08-26"
date-updated: "2026-08-26"
date-review: "2027-02-26"
date-triggers: ["2026-11-26"]
version: "0.1.0"
status: "proposed"
aliases: []
tags: [doc/architecture/gdr]
supersedes: []
superseded-by: []
related-to: ["adr-202608021744-sglang-glom-runtime-mapping", "adr-20260831-mac-studio-multi-host-deployment"]
scope:
  impact-scope:
    - "model-*.sh runner scripts (new model-freetoken-*.sh pattern)"
    - "runner-lib.sh shared library (add FreeToken launch function)"
    - "ADR-202608021744 tiered runtime mapping (Colibri → FreeToken)"
    - "Host port allocation (8004+ for FreeToken instances)"
    - "Multi-agent orchestration layer (ft launch integration)"
    - "Frontier MoE model serving (FreeToken hybrid backend replaces Colibri disk-streaming)"
  excluded-scope:
    - "Dense model serving (stays on SGLang)"
    - "GPTQ model serving (stays on vLLM)"
    - "Code execution sandbox (stays on Glom)"
    - "Model weights and quantization format selection (except GPTQ→NVFP4 migration evaluation)"
hardware:
  target: "NVIDIA DGX Spark (Blackwell GB10 128GB / GB20)"
  container: "Native Python install (uv pip install freetoken[accel]) — not Docker-based"
  validated-precisions: ["NVFP4", "FP8", "MXFP4", "BF16", "GGUF"]
  unsupported-precisions: ["GPTQ", "AWQ (via GPTQ kernel)", "compressed-tensors"]
  frontier-engine: "FreeToken (FlashML-org/FreeToken) — replaces Colibri"
---

# ADR Amendment: Add FreeToken as MoE Serving Runtime, Replace Colibri

**Filename:** `adr-20260826-freetoken-runtime-amendment.md`

- belongs in `internal-docs/adr/2026/08/`
- amends [ADR-202608021744](adr-202608021744-sglang-glom-runtime-mapping.md)

---

## Context

ADR-202608021744 established a tiered runtime selection for the DGX Spark:
SGLang (default for in-VRAM), vLLM (fallback for GPTQ/parallelism), llama.cpp
(new-arch/GGUF), Colibri (frontier MoE disk-streaming), and Glom (code
execution).

On Aug 17 2026, FlashML released **FreeToken**
([arXiv:2608.16157](https://arxiv.org/abs/2608.16157),
[GitHub](https://github.com/FlashML-org/FreeToken)), an edge-native MoE
serving engine co-authored by Matei Zaharia and Ion Stoica. FreeToken
overlaps with two slots in the existing ADR:

1. **Colibri's slot** (frontier MoE that doesn't fit in VRAM) — FreeToken is
   a direct, more sophisticated replacement.
2. **SGLang's slot for MoE models** (in-VRAM MoE with multi-agent workloads) —
   FreeToken's semantic-aware agentic caching and elastic memory management
   are purpose-built for this scenario, which SGLang addresses only through
   general-purpose RadixAttention.

FreeToken does **not** overlap with:
- SGLang's dense-model slot (FreeToken is MoE-focused)
- vLLM's GPTQ slot (FreeToken does not support GPTQ)
- llama.cpp's new-arch/GGUF slot (FreeToken has narrower architecture support)
- Glom's code-execution slot

See the research note
[`note-freetoken-edge-native-moe-serving.md`](../../note-freetoken-edge-native-moe-serving.md)
for the full analysis.

## Decision

Amend the ADR-202608021744 runtime mapping:

### Updated Scenario-to-Runtime Mapping

| Scenario | Previous Default | New Default | Fallback |
|----------|:----------------:|:-----------:|:--------:|
| In-VRAM **MoE**, multi-agent long-context | SGLang | **FreeToken** | SGLang |
| In-VRAM **dense** models, multi-agent | SGLang | SGLang (unchanged) | vLLM |
| GPTQ / compressed-tensors / multi-GPU PP | vLLM | vLLM (unchanged) | — |
| Frontier MoE that doesn't fit in VRAM (GLM-5.2 753B) | Colibri | **FreeToken** (hybrid backend) | Colibri (retained as emergency fallback) |
| Brand-new architecture / GGUF-only | llama.cpp | llama.cpp (unchanged) | — |
| Code execution | Glom | Glom (unchanged) | — |

### Updated Model-Type-to-Runtime Matrix

| Model Type | FreeToken | SGLang | vLLM | llama.cpp | Glom |
|------------|:---------:|:------:|:----:|:---------:|:----:|
| General (dense) | no | **default** | fallback | if too new | no |
| Chat (dense) | no | **default** | fallback | if too new | no |
| Reasoning (MoE) | **default** | fallback | if GPTQ | no | no |
| Code (MoE) | **default** (generate) | fallback (generate) | if GPTQ | no | yes (execute) |
| Frontier MoE (GLM-5.2) | **default** | no | no | no | no |

### What Changes

1. **Colibri → FreeToken** for frontier MoE. Colibri is retained as an
   emergency fallback but is no longer the default. FreeToken's bandwidth-
   adaptive `hybrid` backend, semantic-aware caching, elastic memory, and
   native agent integration make it strictly stronger for this scenario class.

2. **SGLang → FreeToken** for in-VRAM MoE models. SGLang is retained as the
   fallback and remains the default for dense models. The split is: FreeToken
   for MoE, SGLang for dense — both are in-VRAM, but the workload shapes
   differ (MoE benefits from expert caching + bandwidth adaptation; dense
   benefits from SGLang's broader architecture support and production miles).

3. **vLLM unchanged** for GPTQ. The user's current
   `Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-int4` stays on
   vLLM until a NVFP4 re-quantization of the distilled model is produced and
   validated.

4. **New host port range**: FreeToken instances use 8004+ (8000-8003 are
   taken by the existing SGLang/vLLM scripts). FreeToken's default port is
   1919; override with `--port 8004` to stay within repo convention.

### What Does NOT Change

- SGLang remains the default for dense models (Qwen3-Next-80B, etc.)
- vLLM remains the fallback for GPTQ, compressed-tensors, multi-GPU PP
- llama.cpp remains the first-mover for brand-new architectures and GGUF
- Glom remains the only code-execution sandbox
- The `runner-lib.sh` dispatcher pattern is preserved — a new
  `run_freetoken_container()` (or `run_freetoken()` since FreeToken is
  native Python, not Docker) is added alongside the existing SGLang/vLLM
  functions

## Rationale

### Why FreeToken Replaces Colibri

| Dimension | Colibri | FreeToken |
|-----------|---------|-----------|
| Architecture | Single C program, disk-streaming | Python + CUDA kernels, bandwidth-adaptive |
| MoE backend | Disk-stream only | fused / offload / cpu / hybrid (auto-calibrated) |
| Agentic caching | None | Semantic anchor checkpoints (tool calls, thinking blocks) |
| Memory management | Static | Elastic VRAM re-allocation without restart |
| Agent integration | OpenAI API only | `ft launch claude/codex/opencode` + OpenAI + Anthropic APIs |
| Model support | GLM-5.2 only | 20+ MoE models (DeepSeek-V4, Qwen3.5/3.6, GLM-5.2, gpt-oss, Gemma-4, ...) |
| Team | JustVugg (small community) | Yang, Fan, Pan, Xi, Wang, Sun, Keutzer, Han, Zaharia, Xu, Stoica |
| Stars | — | 8.3k in ~9 days |
| License | — | Apache 2.0 |

FreeToken is strictly stronger in every dimension that matters for this repo's
workload. Colibri is retained only as an emergency fallback in case FreeToken
has a critical bug on the GB10.

### Why FreeToken Augments (Not Replaces) SGLang for In-VRAM MoE

- **FreeToken's advantage for MoE**: semantic-aware caching for agent loops,
  elastic memory, bandwidth-adaptive fallback (won't OOM — falls back to
  hybrid), native `ft launch` agent integration.
- **SGLang's advantage for dense**: broader architecture support, cu130-native
  container, more production miles, speculative decoding (DFlash, Spec V2,
  MTP), multi-LoRA serving.
- **The honest split**: MoE → FreeToken, dense → SGLang. Both are in-VRAM;
  the workload shapes differ enough to justify two runtimes.

### Why vLLM Stays for GPTQ

FreeToken supports MXFP4, NVFP4, FP8, BF16, and GGUF — **not GPTQ**. The
user's current reasoning model
(`codgician/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-int4`)
is a GPTQ-int4 checkpoint. Until a NVFP4 re-quantization of the Claude-4.6-Opus
reasoning distillation is produced and validated, this model stays on vLLM.
This is the documented fallback, not a failure.

## Technical Approach

### FreeToken is Native Python, Not Docker

Unlike SGLang and vLLM (which run in Docker containers), FreeToken is
installed as a Python package:

```bash
uv pip install "freetoken[accel]"
```

This means the `runner-lib.sh` dispatcher needs a new `run_freetoken()`
function that does **not** use `docker run`. Instead it launches the FreeToken
server directly on the host:

```bash
run_freetoken() {
  local model="$1"
  local host_port="$2"
  shift 2
  local additional_args=("$@")

  echo "Starting FreeToken server for model: $model"
  echo "Host Port: $host_port"

  ft serve --model "$model" --port "$host_port" "${additional_args[@]}"
}
```

This is simpler than the Docker-based runners but means FreeToken runs outside
the container isolation boundary. The trade-off is acceptable for a
single-operator playground.

### New Script Pattern: `model-freetoken-*.sh`

Following the existing `model-*.sh` pattern:

```bash
#!/usr/bin/env bash
#
# FreeToken MoE Model Runner
# Model: Qwen3.5-35B-A3B (base, NVFP4)
# Runtime: FreeToken — MoE serving with bandwidth-adaptive execution
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/runner-lib.sh"

RUNTIME=freetoken

ACCT="${ACCT:-nvidia}"
MODEL="${MODEL:-Qwen3.6-35B-A3B-NVFP4}"
HOST_PORT="${HOST_PORT:-8004}"

main() {
  load_env
  check_prerequisites
  setup_huggingface
  stop_existing_container "$MODEL"

  run_freetoken "$ACCT/$MODEL" "$HOST_PORT"
}

main "$@"
```

### Frontier MoE: GLM-5.2 753B on FreeToken

```bash
ft serve --model nvidia/GLM-5.2-NVFP4 --port 8005 --moe-backend hybrid
```

Run `ft bench bw` once to calibrate the hybrid split for the GB10's
VRAM-to-host-RAM bandwidth. FreeToken's `hybrid` backend dynamically splits
each step between PCIe fetch and CPU compute — the calibrated profile tells
it the optimal split for this specific hardware.

### Agent Integration

```bash
# Point Claude Code at the FreeToken server
ft launch claude --server http://127.0.0.1:8004

# Or manually configure any OpenAI-compatible client
# Base URL: http://127.0.0.1:8004/v1
```

## Affected Components

- **`runner-lib.sh`** — add `run_freetoken()` function and `RUNTIME=freetoken`
  dispatch case. FreeToken is native Python (no Docker), so the function is
  simpler than the SGLang/vLLM runners.
- **New `model-freetoken-*.sh` scripts** — one per MoE model served by
  FreeToken. Initial set: `model-freetoken-reason.sh` (Qwen3.5-35B-A3B NVFP4,
  port 8004), `model-freetoken-frontier.sh` (GLM-5.2 NVFP4, port 8005).
- **`model-reason.sh`** — stays on vLLM (GPTQ-int4) until NVFP4 re-quantization
  is done. Add a comment pointing to this ADR.
- **ADR-202608021744** — update the scenario-to-runtime table and model-type
  matrix to reflect FreeToken replacing Colibri and augmenting SGLang for MoE.
- **README model table** — add FreeToken rows with the "Runtime" column.
- **`internal-docs/oos/`** — Colibri moves from "active runtime" to
  "emergency fallback only."

## Consequences

### Positive

- **Frontier MoE serving is agent-native.** FreeToken's `ft launch` and
  semantic-aware caching make GLM-5.2 753B directly usable by Claude Code,
  Codex, and OpenCode — not just a raw OpenAI endpoint.
- **In-VRAM MoE gets agentic caching.** Multi-turn agent loops with tool
  calls and thinking blocks reuse KV cache instead of recomputing. Direct
  throughput win for the 32-128 concurrent agent workload.
- **No OOM cliff.** FreeToken's bandwidth-adaptive fallback means a model
  that doesn't quite fit at the desired context length transparently falls
  back to `hybrid` mode instead of crashing. SGLang has no such fallback.
- **Elastic memory.** Dynamic VRAM re-allocation between expert caches and
  KV memory without restarts — no more manual `--mem-fraction-static` tuning
  when context length changes.
- **Colibri dependency removed.** One fewer C engine to build, pin, and
  maintain on the host.

### Negative

- **One more runtime.** The operator now manages SGLang (dense), FreeToken
  (MoE), vLLM (GPTQ fallback), and llama.cpp (new-arch). The operational
  surface grows — but FreeToken replaces Colibri, so it is net-neutral on
  runtime count.
- **FreeToken is native Python, not Docker.** It runs outside the container
  isolation boundary. For a single-operator playground this is acceptable,
  but it means FreeToken does not get Docker's resource isolation.
- **GPTQ models cannot migrate yet.** The user's Claude-4.6-Opus reasoning
  distillation stays on vLLM until a NVFP4 re-quantization is produced.
- **GB10 / SM103 support is unverified.** FreeToken's README lists RTX
  30/40/50 series. The GB10 is Blackwell (SM103). Needs validation —
  FreeToken uses CUDA kernels that may or may not target SM103 natively.
- **FreeToken is very new** (paper Aug 17 2026, ~9 days old as of this ADR).
  Production maturity is unknown. SGLang has 400k+ GPUs in production;
  FreeToken has 8.3k GitHub stars and no documented production deployments
  yet.

### Neutral

- **Host port allocation extends to 8004+** for FreeToken instances.
  Existing 8000-8003 scripts are unchanged.
- **`.env` and `HUGGING_FACE_HUB_TOKEN`** — FreeToken reads the same HF
  token for model pulls.

## Rollout

1. **Validate GB10 support.** Install FreeToken on the DGX Spark, run
   `ft bench bw`, and serve `nvidia/Qwen3.6-35B-A3B-NVFP4` on port 8004.
   Confirm the server starts and responds. If SM103 is not supported,
   FreeToken stays proposed-not-accepted and Colibri retains the frontier
   MoE slot.
2. **Add `run_freetoken()` to `runner-lib.sh`** with the native Python launch
   path (no Docker).
3. **Create `model-freetoken-reason.sh`** (Qwen3.5-35B-A3B NVFP4, port 8004)
   as the canary. Smoke-test with `test-query.sh -p 8004`.
4. **Benchmark FreeToken vs SGLang** for Qwen3.5-35B-A3B at 32-128 concurrent
   agents, 64k-256k context. Measure: tokens/sec, TTFT, cache hit rate,
   inter-token latency. If FreeToken wins on the agentic workload (multi-turn
   with tool calls), it becomes the MoE default. If SGLang wins, FreeToken
   stays as the frontier MoE runtime only (replacing Colibri) and SGLang
   retains the in-VRAM MoE slot.
5. **Create `model-freetoken-frontier.sh`** (GLM-5.2 NVFP4, port 8005) with
   `--moe-backend hybrid`. Run `ft bench bw` first to calibrate.
6. **Update ADR-202608021744** — apply the scenario-to-runtime and
   model-type matrix changes from this amendment.
7. **Update README model table** — add FreeToken rows, add "Runtime" column.
8. **Retire Colibri** — remove Colibri from the active runtime list, move it
   to `internal-docs/oos/` as an emergency fallback. Remove the Colibri
   install/pin docs from the ADR.
9. **Evaluate NVFP4 re-quantization** of the Claude-4.6-Opus reasoning
   distillation. If successful, migrate `model-reason.sh` from vLLM/GPTQ to
   FreeToken/NVFP4. If not, GPTQ stays on vLLM indefinitely.

## Alternatives Considered

**Option A — Keep Colibri, do not adopt FreeToken.**
Pros: zero change. Cons: misses semantic-aware caching for agent loops;
Colibri is a smaller community project with narrower model support; FreeToken
is strictly stronger for the same scenario class.

**Option B — FreeToken replaces both Colibri AND SGLang for all MoE.**
Pros: simplest mental model (one MoE runtime). Cons: removes SGLang as the
documented fallback for in-VRAM MoE; if FreeToken has a critical bug on GB10,
there is no fallback. Rejected — SGLang must be retained as fallback.

**Option C (chosen) — FreeToken replaces Colibri, augments SGLang for MoE.**
Pros: correct scenario split (MoE → FreeToken, dense → SGLang); SGLang
retained as fallback; Colibri retired. Cons: one more runtime to manage
(net-neutral since Colibri is removed); FreeToken is very new.

**Option D — Wait for FreeToken to mature before adopting.**
Pros: avoids early-adopter risk. Cons: the semantic-aware caching and
bandwidth-adaptive execution are available now and directly address this
repo's workload; waiting has an opportunity cost in throughput.

## To Investigate

- **GB10 / SM103 support**: does FreeToken's CUDA kernel cache include
  prebuilt kernels for SM103, or does it JIT-compile? Run `ft serve` on the
  DGX Spark and check for kernel compilation errors.
- **FreeToken + Docker**: can FreeToken be containerized for isolation, or
  does it need direct host access to the GPU and host RAM for the hybrid
  backend? If containerizable, add a `run_freetoken_container()` function.
- **NVFP4 re-quantization pipeline**: what tools produce NVFP4 checkpoints
  from a BF16 or GPTQ source? NVIDIA's TensorRT Model Optimizer? FreeToken's
  `ft checkpoint`? Can the Jackrong/Codgician distillation be converted?
- **Semantic caching vs RadixAttention**: is FreeToken's semantic anchor
  checkpointing a superset of SGLang's RadixAttention, or do they address
  different reuse patterns? If RadixAttention already captures the agent-loop
  prefix reuse, FreeToken's advantage may be smaller than expected.
- **FreeToken + Devin/Cursor**: does `ft launch` conflict with existing
  provider configs for Devin, Cursor, or Continue? Can FreeToken serve as
  the backend for Cursor's local-model autocomplete via Continue?
- **FreeToken + SGLang coexistence**: can FreeToken and SGLang run
  simultaneously on the GB10 (FreeToken for MoE on port 8004, SGLang for
  dense on port 8000)? VRAM partitioning needs validation.

## Validation

- [ ] FreeToken installs and starts on the DGX Spark without kernel errors
- [ ] `ft bench bw` completes and produces a bandwidth profile for the GB10
- [ ] `ft serve --model nvidia/Qwen3.6-35B-A3B-NVFP4 --port 8004` responds to
      `/v1/chat/completions` with correct output
- [ ] `ft serve --model nvidia/GLM-5.2-NVFP4 --port 8005 --moe-backend hybrid`
      serves GLM-5.2 753B without OOM
- [ ] Benchmark: FreeToken vs SGLang for Qwen3.5-35B-A3B at 32 concurrent
      agents, 64k context — FreeToken wins on throughput or cache hit rate
- [ ] `test-query.sh -p 8004` returns HTTP 200
- [ ] No host port collision with existing 8000-8003 scripts
- [ ] ADR-202608021744 updated with the new scenario-to-runtime mapping
