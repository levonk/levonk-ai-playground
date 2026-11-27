---
modeline: "vim: set ft=markdown:"
title: "ADR Amendment: Add Project Maya as Model-Specific Preferred Engine for Its Supported MoE Families"
gdr-id: "gdr20261027001"
slug: "project-maya-model-specific-engine"
url: "https://github.com/levonk/levonk-ai-playground/blob/main/internal-docs/adr/2026/10/adr-20261027-project-maya-model-specific-engine.md"
synopsis: "Amends ADR-202608021744 (as amended by ADR-20260826) to add Project Maya (mw00/project-maya) as the preferred engine for the MoE families it supports — GLM-5.3 744B, GLM-5.3-Flash 321B, DeepSeek-V4.1-Flash-1017 748B, MiniMax-M3.1-Flash 428B, Hy4-Flash 220B, Ling-3.2-air 128B, and Inkling 2 Mini 26B (grew from 3 families at drafting to 7 by 2026-11-27) — pending a head-to-head GB10 benchmark against FreeToken's hybrid backend. FreeToken remains the general MoE default (in-VRAM and frontier); Colibri remains the emergency frontier fallback; SGLang/vLLM/llama.cpp/Glom slots are unchanged. Maya is a specialist product (per-model engines, own quant tiers, dashboard, OpenAI+Anthropic APIs), not a general MoE runtime — adoption is conditional on the benchmark and on resolving its RAM-pinning and 8-request concurrency ceiling for the multi-agent workload."
author: "https://github.com/levonk"
date-created: "2026-10-27"
date-updated: "2026-11-27"
date-review: "2027-04-27"
date-triggers: ["2027-01-27"]
version: "0.2.0"
status: "proposed"
aliases: []
tags: [doc/architecture/gdr]
supersedes: []
superseded-by: []
related-to: ["adr-202608021744-sglang-glom-runtime-mapping", "adr-20260826-freetoken-runtime-amendment", "adr-20260831-mac-studio-multi-host-deployment"]
scope:
  impact-scope:
    - "Frontier/mid-size MoE serving for Maya's supported families: GLM-5.3 744B, GLM-5.3-Flash 321B, DeepSeek-V4.1-Flash-1017 748B, MiniMax-M3.1-Flash 428B, Hy4-Flash 220B, Ling-3.2-air 128B, Inkling 2 Mini 26B"
    - "model-*.sh runner scripts (new model-maya-*.sh pattern)"
    - "runner-lib.sh shared library (add Maya launch function — native setup.sh build, not Docker)"
    - "Host port allocation (8006+ for Maya instances; Maya defaults to 8080)"
    - "ADR-202608021744 tiered runtime mapping (frontier-MoE row gains a model-specific preferred engine)"
  excluded-scope:
    - "General MoE serving — FreeToken remains the default for all MoE checkpoints outside Maya's supported families"
    - "Dense model serving (stays on SGLang)"
    - "GPTQ model serving (stays on vLLM)"
    - "Code execution sandbox (stays on Glom)"
    - "Multi-agent orchestration layer — Maya's 8-request parallel cap does not cover the 32-128 agent target; fleet workloads stay on FreeToken/SGLang"
hardware:
  target: "NVIDIA DGX Spark (Blackwell GB10 128GB / GB20)"
  container: "None — Project Maya builds CUDA locally via ./setup.sh (nvcc + NVTX/profiler headers; CUDA 12.8+ for SM103)"
  validated-precisions: ["Maya GGUF tiers (XS/S/M/L, ~1.6-4 bit experts)", "EXL3", "community GGUF"]
  frontier-engine-model-specific: "Project Maya (mw00/project-maya) — pending GB10 benchmark vs FreeToken hybrid"
  frontier-engine-general: "FreeToken (FlashML-org/FreeToken) — unchanged general default"
  frontier-engine-emergency: "Colibri (JustVugg/colibri) — unchanged emergency fallback"
---

# ADR Amendment: Add Project Maya as Model-Specific Preferred Engine for Its Supported MoE Families

**Filename:** `adr-20261027-project-maya-model-specific-engine.md`

- belongs in `internal-docs/adr/2026/10/`
- amends [ADR-202608021744](../08/adr-202608021744-sglang-glom-runtime-mapping.md)
  as amended by [ADR-20260826](../08/adr-20260826-freetoken-runtime-amendment.md)

---

## Context

ADR-202608021744 established the tiered runtime selection for the DGX Spark,
and ADR-20260826 amended it to make FreeToken the default for all MoE serving
(both in-VRAM and frontier) with Colibri demoted to emergency fallback.

On Oct 6 2026, **Project Maya** ([mw00/project-maya](https://github.com/mw00/project-maya),
MIT) was released — a packaged frontier-MoE serving product (engine + server +
dashboard + its own quantized weights), started from
[Strata](https://github.com/Niko1221/Strata). At drafting it supported three
model families; as of 2026-11-27 it supports **seven**, spanning 26B–744B:

- **GLM-5.3** (744B, 40B active, 1M ctx, text-only) — added post-drafting
- **GLM-5.3-Flash** (321B, ~18B active, 1M ctx)
- **DeepSeek-V4.1-Flash-1017** (748B)
- **MiniMax-M3.1-Flash** (428B, 23B active, 1M ctx)
- **Hy4-Flash** (Tencent, 220B, vision, NVIDIA + AMD-on-Linux) — added post-drafting
- **Ling-3.2-air** (inclusionAI, 128B, vision, MTP) — added post-drafting
- **Inkling 2 Mini** (Thinking Machines, 26B, vision) — added post-drafting

Each family gets a purpose-built CUDA engine (`maya`, `maya-ds4`,
`maya-mm3`, …) with three-tier expert placement (VRAM → RAM → NVMe), its
own measured quant tiers (XS/S/M/L), a guided `./setup.sh` that auto-tunes
to the machine, a web dashboard, and OpenAI + Anthropic + Responses APIs.
Supported from **8 GB of VRAM** on NVIDIA (V100/RTX 20+) and AMD on Linux;
Windows 10/11 supported (`START-MAYA.bat`); Apple Silicon Metal backend is
experimental (GLM-5.3-Flash only).

Maya overlaps with exactly one slot in the existing mapping — the
**frontier-MoE slot** — and only partially: it covers seven curated
families, where FreeToken covers 20+ MoE checkpoints. It does **not**
overlap with:

- SGLang's dense-model slot (Maya is MoE-only, curated families)
- FreeToken's in-VRAM MoE slot for general checkpoints (no coverage outside
  its supported families; no semantic anchor caching)
- vLLM's GPTQ slot
- llama.cpp's new-arch/GGUF slot (Maya consumes only its own GGUF builds)
- Glom's code-execution slot

Two properties make Maya different from every other engine in the tier map:
it ships **per-model engines** rather than a generic runtime, and it ships
**its own quant tiers with published accuracy deltas** (97.7–99.2% of the
FP8 originals' zero-shot accuracy on GLM-5.3-Flash; similar published
numbers for Hy4-Flash and Ling-3.2-air) — the quant work that FreeToken's
GPTQ-gap forces onto the operator is already done for Maya's models.

**Lineage note (added 2026-11-27):** Maya's upstream, Strata, shipped as a
packaged consumer product in its own right (0.2.x line: one-click
Windows/Linux install, tuned SQ/GSQ-RCO quant packs, DFlash2 drafter,
OpenAI+Anthropic APIs). Strata's model family (Qwen3.8-Flash-Next,
Qwen3.6-35B-A3B, HY-4 Flash, Inkling 2 Mini) overlaps Maya's on Hy4-Flash
and Inkling 2 Mini — for those two, the engine choice is between Strata's
packaged app and Maya's frontier-scale engines. Strata is also the
measured-fastest GGUF path on the GB10 (sm_121) for its own packs per the
vault matrix; it does not serve frontier-size models and does not compete
for this ADR's frontier slot.

See the research note
[`note-project-maya-frontier-moe-serving.md`](../../../note-project-maya-frontier-moe-serving.md)
for the full analysis, and the vault-side landscape matrix
(`project-comparison-local-llm-model-runners-20261027-115706.md`) for the
wider runner comparison.

## Decision

Amend the ADR-202608021744 runtime mapping **for the frontier-MoE row only**:

### Updated Scenario-to-Runtime Mapping

| Scenario | Previous Default | New Default | Fallback |
|----------|:----------------:|:-----------:|:--------:|
| In-VRAM **MoE**, multi-agent long-context | FreeToken | FreeToken (unchanged) | SGLang |
| In-VRAM **dense** models, multi-agent | SGLang | SGLang (unchanged) | vLLM |
| GPTQ / compressed-tensors / multi-GPU PP | vLLM | vLLM (unchanged) | — |
| Frontier MoE that doesn't fit in VRAM | FreeToken (hybrid) | FreeToken — **general default**; **Project Maya — preferred for its supported families (GLM-5.3/5.3-Flash, DS-V4.1, MM3.1, Hy4, Ling-3.2-air, Inkling 2 Mini), pending GB10 benchmark** | Colibri (emergency); Remote API (temporary, violates self-hosted) |
| Brand-new architecture / GGUF-only | llama.cpp | llama.cpp (unchanged) | — |
| Code execution | Glom | Glom (unchanged) | — |

### Updated Model-Type-to-Runtime Matrix

| Model Type | FreeToken | Maya | SGLang | vLLM | llama.cpp | Colibri | Glom |
|------------|:---------:|:----:|:------:|:----:|:---------:|:-------:|:----:|
| General (dense) | no | no | **default** | fallback | if too new | no | no |
| Chat (dense) | no | no | **default** | fallback | if too new | no | no |
| Reasoning (MoE) | **default** | no | fallback | if GPTQ | no | no | no |
| Code (MoE) | **default** (generate) | no | fallback (generate) | if GPTQ | no | no | yes (execute) |
| Frontier MoE — Maya's supported families (GLM-5.3/5.3-Flash, DS-V4.1, MM3.1, Hy4, Ling-3.2-air, Inkling 2 Mini) | fallback | **preferred pending benchmark** | no | no | no | emergency | no |
| Frontier MoE — all other checkpoints | **default** | n/a | no | no | no | emergency | no |

### What Changes

1. **Frontier-MoE row splits into model-specific and general.** For the
   families Maya supports (seven as of 2026-11-27), Maya is the preferred
   engine *pending a GB10 head-to-head benchmark* against FreeToken
   `hybrid` (see Rollout). For every other MoE checkpoint, FreeToken
   remains the default.
2. **Maya is native + self-built, like FreeToken but more so.** No Docker:
   `./setup.sh` compiles CUDA locally (needs nvcc, NVTX/profiler headers;
   CUDA 12.8+ for SM103/Blackwell). A `run_maya()` function joins
   `runner-lib.sh` alongside `run_freetoken()`.
3. **New host port range**: Maya defaults to 8080; this repo assigns 8006+
   (`./maya.sh --setup --port 8006`) to stay within the 8000-series
   convention (8000–8003 SGLang/vLLM, 8004–8005 FreeToken).
4. **Colibri unchanged** — still the emergency fallback if both FreeToken
   and Maya fail on GB10.

### What Does NOT Change

- FreeToken remains the default for **all** in-VRAM MoE and for frontier
  MoE outside Maya's supported families — Maya does not cover general
  checkpoints and has no semantic anchor caching.
- SGLang remains the default for dense models.
- vLLM remains the fallback for GPTQ, compressed-tensors, multi-GPU PP —
  Maya reads neither GPTQ nor HF safetensors for arbitrary models.
- llama.cpp remains the first-mover for new architectures and GGUF-only
  quants — Maya's GGUF support is limited to its own published packs.
- Glom remains the only code-execution sandbox.
- The `runner-lib.sh` dispatcher pattern is preserved.

## Rationale

### Why Maya Is Preferred for Its Supported Families (Pending Benchmark)

| Dimension | FreeToken (general MoE) | Project Maya (7 families) |
|-----------|------------------------|---------------------------|
| Engine design | One generic engine, many checkpoints | Purpose-built engine per model family |
| Quant supply | Operator supplies checkpoint (MXFP4/NVFP4/FP8/BF16/GGUF) | Own measured XS/S/M/L tiers + EXL3/GGUF options; setup picks for your memory pool |
| Quant quality evidence | None published per-model | Published vs FP8/original: 97.7–99.2% zero-shot (GLM-5.3), 88.5–96.6% (Hy4), 88.1–93.8% (Ling-3.2-air) token agreement |
| Offloading | Bandwidth-adaptive q* (fused/offload/cpu/hybrid) | 3-tier VRAM→RAM→NVMe hot-expert placement + `--calibrate` |
| Agentic caching | 🏆 Semantic anchor checkpoints | DS4 conversation slots (parking/swap, DeepSeek engine only) |
| Agent integration | `ft launch` + OpenAI + Anthropic APIs | OpenAI + Anthropic + Responses APIs, streamed tool calls, thinking levels |
| Concurrency | Agent-fleet oriented | `--parallel` ≤ 8 sessions |
| UI | Desktop app | Built-in chat + live per-GPU hardware monitor |
| Model coverage | 20+ MoE checkpoints | 7 families — hard boundary |
| GB10 evidence | SM103 unverified (open question in ADR-20260826) | Published user report: 16.5 tok/s decode @128K on DGX Spark |
| Maturity | ~6 months, 17.8k★, strong team | ~8 weeks, 3.4k★, lead + co-maintainer + contributors |

For the covered families, Maya's specialization advantage is real:
the engine knows the prompt format, MTP block, drafter, vision tower, and
thinking levels natively, and the quant tiers are calibrated per memory
pool — the operator does no quantization work. The coverage gap is
narrowing (3 → 7 families in 8 weeks, now spanning 26B–744B), which
weakens — but does not eliminate — FreeToken's breadth argument.

### Why FreeToken Stays the General Default

- **Coverage**: 20+ checkpoints vs 7 families. Model rotation depends on
  breadth.
- **Agentic caching**: semantic anchor checkpoints are the purpose-built
  feature for the 32–128 concurrent-agent workload. Maya's parallel cap is
  8 sessions; DS4 slots are state parking, not anchor-point reuse.
- **Elastic memory**: FreeToken re-allocates VRAM↔KV without restart.
  Maya pins nearly all free RAM at startup (see Consequences).

### Why Not Make Maya the Frontier Default Outright

- ~8 weeks old, 3.4k stars, lead maintainer + co-maintainer — vs
  FreeToken's Zaharia/Stoica/Han/Keutzer team and faster accumulation of
  production signal. Bus factor and churn both favor caution.
- The ×8 parallel ceiling contradicts the repo's 32–128 agent target for
  fleet workloads; Maya's sweet spot is a few high-quality sessions.
- RAM pinning makes co-location with the repo's other resident servers
  unproven.

## Technical Approach

### Maya Is a Local CUDA Build, Not Docker, Not pip

```bash
git clone https://github.com/mw00/project-maya.git && cd project-maya
./setup.sh                       # interactive: measures GPU/CPU/RAM/SSD, picks tier + context
./setup.sh --model glm-maya-s --port 8006   # non-interactive
./maya.sh --calibrate            # one-time per-config hardware tuning (10-15 min)
```

Build requirements on GB10: `nvcc` + profiler/NVTX headers (CUDA 12.8+ for
SM103; README documents the `nvidia-cuda-profiler-api`/`nvidia-nvtx` header
wheels for CUDA 13), g++, Python 3.10+.

```bash
run_maya() {
  local model_id="$1"   # e.g. glm-maya-s, deepseek-maya-xs, minimax-maya-s
  local host_port="$2"
  shift 2
  local additional_args=("$@")

  echo "Starting Project Maya for model: $model_id"
  echo "Host Port: $host_port"

  (cd "$MAYA_HOME" && ./maya.sh --model "$model_id" --port "$host_port" "${additional_args[@]}")
}
```

### New Script Pattern: `model-maya-*.sh`

Following the existing `model-*.sh` pattern (same shape as the
`model-freetoken-*.sh` pattern from ADR-20260826), with `RUNTIME=maya`,
`MAYA_HOME` pointing at the checkout, and `HOST_PORT=8006`.

### Tier Selection on GB10 (121 GB Unified Memory)

- **GLM-5.3-Flash**: `glm-maya-s` (96.5 GB) — fits mostly in unified
  memory; published Spark number exists. `glm-maya-s24` is tuned for ≤24
  GB cards and is likely the wrong pick on unified memory; `glm-maya-m`
  (116 GB) is the quality alternative if the pool allows.
- **Hy4-Flash**: `hy4-flash-maya-xs` (44.8 GB) or `-s` (53.0 GB) — fits
  comfortably; best vision-capable MoE option on the box (+63.4 GB n-gram
  table on SSD).
- **Ling-3.2-air**: `ling-3.2-air-maya-s` (36.9 GB) — the easiest fit of
  the family list; good candidate for the *first* GB10 validation run.
- **Inkling 2 Mini**: `inkling-2-mini` (17.6 GB) — trivially fits; the
  only Maya model that leaves room for co-resident servers.
- **MiniMax-M3.1**: `minimax-maya-xs` (93 GB) fits; `maya-s` (122 GB) is
  marginal.
- **DeepSeek-V4.1**: `deepseek-maya-xs` (277 GB) — streams substantially
  from NVMe even on 128 GB; expect the lowest tokens/s of the list.
- **GLM-5.3 (744B)**: `glm-5.3-maya-xs` (170 GB) — needs a ~192 GB total
  memory pool; beyond GB10 unless other RAM is added. Documented as
  unsupported on this box for now.

## Affected Components

- **`runner-lib.sh`** — add `run_maya()` and `RUNTIME=maya` dispatch.
- **New `model-maya-*.sh` scripts** — initial set:
  `model-maya-glm.sh` (glm-maya-s, port 8006). `model-maya-ling.sh`
  (ling-3.2-air-maya-s, 36.9 GB) and `model-maya-hy4.sh`
  (hy4-flash-maya-xs, 44.8 GB) are strong second candidates — they fit
  GB10 far more comfortably and de-risk the build/co-location checks
  before the big models. DeepSeek/MiniMax scripts follow if the
  benchmark justifies them.
- **ADR-202608021744** — frontier-MoE row gains the model-specific
  preferred engine (this amendment supplies the text).
- **README model table** — add a Maya row with the "Runtime" column.
- **`internal-docs/platforms/nvidia-gb10-128gb.md`** — add Maya build notes
  (CUDA 12.8+/13 headers, RAM pinning caveat).

## Consequences

### Positive

- **Frontier MoE on GB10 becomes turnkey for the biggest open MoE
  families** — one command, auto-tuned, quant quality pre-measured. The
  operator does no GGUF/EXL3 selection work.
- **The only runner with a published DGX Spark number** (16.5 tok/s
  GLM-5.3 Maya-S @128K) — derisks SM103 support ahead of FreeToken's
  still-unverified status.
- **Anthropic endpoint native** — Claude Code points at
  `ANTHROPIC_BASE_URL` with no adapter.
- **Quant quality is published, not assumed** — 97.7–99.2% zero-shot vs
  the FP8 originals, per tier.

### Negative

- **RAM pinning**: Maya pins all but ~6 GB of free RAM for its expert
  tier. On the Spark box, that likely means **exclusive use** — it cannot
  trivially co-reside with SGLang/vLLM/FreeToken servers. Operationally,
  Maya sessions may need the box to themselves.
- **Concurrency ceiling**: `--parallel` maxes at 8; the repo's fleet
  target is 32–128. Maya covers the "few deep sessions" workload, not the
  agent fleet.
- **Very young**: released 2026-10-06 (~8 weeks as of 2026-11-27), lead
  maintainer + co-maintainer, 275 open issues, 3.4k★. Fast-moving tracker —
  pin the checkout, treat `--bench`/`--report` output as mandatory
  evidence. The 3→7-family growth in 8 weeks cuts both ways: more
  coverage, but a fast-moving surface to re-validate each update.
- **Build toolchain dependency**: local CUDA compile on every update —
  unlike Docker images, kernel/header drift can break builds.
- **DFlash2 drafter is CC BY-NC-ND** (opt-in only) — excluded from any
  commercial-facing use.

### Neutral

- Host port allocation extends to 8006+ for Maya instances.
- `.env`/`HUGGING_FACE_HUB_TOKEN` reused for gated community quants
  (abliterated variants).

## Rollout

1. **Build check on GB10.** Clone, run `./setup.sh --model glm-maya-s
   --port 8006`. Confirm the CUDA build completes against the installed
   toolkit (12.8+; fetch header wheels if on CUDA 13 split toolkit). If
   SM103 build fails, this amendment stays proposed-not-accepted and
   FreeToken/Colibri retain the slot.
2. **RAM co-location check.** With the existing servers running, start
   Maya and measure remaining free RAM / whether other servers OOM. Decide:
   exclusive-box mode (stop others first) or bounded (`--vram-reserve-mib`
   and equivalents).
3. **Head-to-head benchmark.** Maya `glm-maya-s` vs FreeToken `hybrid` on
   a GLM-5.3-class checkpoint: decode tok/s, prefill on 8K/64K/128K
   prompts, TTFT, and 4-8 concurrent sessions (Maya's ceiling). Run
   `./maya.sh --bench && ./maya.sh --report` and attach the report to the
   tracking issue.
4. **Add `run_maya()` + `model-maya-glm.sh`** (port 8006). Smoke-test with
   `test-query.sh -p 8006`.
5. **Update ADR-202608021744** — apply the frontier-MoE row change.
6. **Update README model table** — Maya row, Runtime column.
7. **Re-evaluate at the 3-month mark** (`date-triggers: 2027-01-27`):
   Maya's issue velocity, maintainer count, and whether the benchmark held.
   If Maya wins decisively and matures, consider promoting it to frontier
   default for its families outright; if it stalls or breaks on GB10,
   revert this row to FreeToken-only.

## Alternatives Considered

**Option A — Do nothing; FreeToken covers frontier MoE.**
Pros: zero change, FreeToken's general coverage. Cons: ignores Maya's
per-model engine + measured quant advantage on the biggest open MoE
families, and Maya's published GB10 number — the only verified Spark data
point in the frontier class.

**Option B — Maya becomes the frontier-MoE default outright.**
Pros: simplest for Maya's supported families. Cons: drops FreeToken's
20+ checkpoint coverage and agentic caching; Maya is ~8 weeks old with an
8-session ceiling and RAM pinning that breaks co-location. Rejected —
too much uncertainty to displace the general default.

**Option C (chosen) — Model-specific preferred engine pending benchmark.**
Pros: captures Maya's specialization where it applies, keeps FreeToken's
breadth and agentic features everywhere else, and makes adoption
evidence-gated (GB10 head-to-head) rather than faith-based. Cons: frontier
row gets slightly more complex; one more native-built runtime to maintain.

**Option D — Wait for Maya to mature before any adoption.**
Pros: avoids early-adopter risk. Cons: same argument rejected in
ADR-20260826 for FreeToken — the capability exists now and the benchmark
is cheap; waiting has an opportunity cost.

## To Investigate

- **SM103/CUDA 13 build**: does `./setup.sh` pick the right toolkit on the
  Spark and complete without header patches? (README documents the
  `CPATH`-targeted header wheel workaround.)
- **RAM pinning bounds**: is there a flag/env to cap the expert tier so
  other servers keep their share? What actually happens to resident
  SGLang/vLLM/FreeToken servers when Maya starts?
- **Benchmark**: Maya vs FreeToken hybrid on the same GLM-5.3-class model
  — decode, prefill, TTFT, multi-session throughput (at Maya's ≤8 cap).
- **DS4 conversation slots vs FreeToken anchor checkpoints** for agent
  parent/worker splits — is parking+spill sufficient, or is anchor-point
  reuse materially faster?
- **Anthropic endpoint fidelity** under real Claude Code traffic: tool
  schemas, long prompt progress streaming, `/v1/messages` edge cases.
- **Maya-M-Derisked** (refusal-reduced weight edit) for the agent fleet's
  refusal-rate problem — evaluate quality and license posture.
- **Co-existence with `ft`/`model-*.sh` lifecycle**: does `setup.sh`'s
  "pins most of free RAM" behavior need a stop-others-first wrapper in
  `runner-lib.sh`?

## Validation

- [ ] `./setup.sh --model glm-maya-s --port 8006` completes on GB10
      (CUDA build succeeds; correct toolkit selected)
- [ ] `curl http://127.0.0.1:8006/v1/chat/completions` returns correct output
- [ ] `./maya.sh --bench` produces decode/prefill numbers on GB10 ≥ the
      published 16.5 tok/s figure (or documented variance)
- [ ] Head-to-head vs FreeToken `hybrid` on same model class recorded and
      attached to the tracking issue
- [ ] RAM co-location behavior documented (exclusive-box required or not)
- [ ] `test-query.sh -p 8006` returns HTTP 200
- [ ] No host port collision with existing 8000–8005 allocations
- [ ] ADR-202608021744 frontier-MoE row updated per this amendment
