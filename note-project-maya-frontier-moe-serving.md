---
title: "Project Maya: Packaged Frontier-MoE Serving — Per-Model Engines with Tiered VRAM/RAM/SSD Expert Placement"
date:
  created: 2026-10-27
  knowledge-basis: 2026-10-27
  last-used: 2026-10-27
tags:
  - research
  - inference-engine
  - moe
  - edge-serving
  - project-maya
  - frontier-models
  - adr-update-candidate
aliases:
  - Project Maya note
  - Maya engine
related:
  - "[[adr-202608021744-sglang-glom-runtime-mapping]]"
  - "[[adr-20260826-freetoken-runtime-amendment]]"
  - "[[adr-20261027-project-maya-model-specific-engine]]"
  - "[[note-freetoken-edge-native-moe-serving]]"
source:
  - "https://github.com/mw00/project-maya"
  - "https://huggingface.co/peasantsmith"
---

# Project Maya: Packaged Frontier-MoE Serving

> Research note — evaluates Project Maya as a contender for the
> frontier-MoE scenario class in
> [[adr-202608021744-sglang-glom-runtime-mapping]], alongside FreeToken
> (default) and Colibri (emergency fallback).

## TL;DR

**Project Maya** (mw00/project-maya, MIT) is a *packaged* frontier-MoE
serving product — engine + server + dashboard + its own quantized weights —
that runs exactly **three model families**: GLM-5.3-Flash (321B),
DeepSeek-V4.1-Flash-1017 (748B), and MiniMax-M3.1-Flash (428B). Each model
gets a purpose-built CUDA engine (`maya`, `maya-ds4`, `maya-mm3`) that
places hot experts in VRAM, warm experts in RAM, and cold experts on NVMe,
with a one-command `./setup.sh` that measures the machine and picks the
quant tier, context length, and drafter.

It is the **strongest direct competitor to the FreeToken/Colibri
frontier-MoE slot — but only for its three supported models**. It is not a
general MoE engine: any checkpoint outside those families simply does not
run. It also lacks FreeToken's semantic-aware agentic caching, which
matters for this repo's 32–128 concurrent-agent workload.

**Bottom line for this repo:** add Maya to the ADR as the **model-specific
preferred engine for GLM-5.3 / DeepSeek-V4.1 / MiniMax-M3.1 pending a GB10
benchmark against FreeToken's `hybrid` backend**. FreeToken stays the
general MoE default; Colibri stays the emergency fallback; nothing else in
the tier map changes. The GB10 data point that already exists (16.5 tok/s
on GLM-5.3 · Maya-S at 128K ctx) is encouraging but needs a head-to-head.

## What Project Maya Is

| Field | Value |
|-------|-------|
| Project | [mw00/project-maya](https://github.com/mw00/project-maya) |
| Stars | 2.1k (as of 2026-10-27; created 2026-10-06 — ~3 weeks old) |
| License | MIT (engine); models keep their own licenses (GLM/DeepSeek MIT; MiniMax Community License) |
| Language | CUDA/C++ — built on [Strata](https://github.com/Niko1221/Strata), [ggml/llama.cpp](https://github.com/ggml-org/llama.cpp), [exllamav3](https://github.com/turboderp-org/exllamav3) EXL3, [beellama.cpp](https://github.com/Anbeeld/beellama.cpp) DFlash2 |
| Install | `git clone` + `./setup.sh` (interactive, auto-tuning); `START-MAYA.bat` on Windows |
| Hardware | NVIDIA CC ≥ 7.0 (V100/RTX 20+), 1–16 GPUs; AMD experimental (RX 7900, R9700, Strix/Gorgon Halo, ROCm 7); Linux x86-64; Windows 10/11 experimental (not WSL2) |
| Models | GLM-5.3-Flash (321B, ~18B active, 1M ctx), DeepSeek-V4.1-Flash-1017 (748B), MiniMax-M3.1-Flash (428B, 23B active, 1M ctx) |

## Key Innovations

### 1. Per-Model Purpose-Built Engines

Unlike FreeToken (one engine, many MoE checkpoints) or llama.cpp (generic
GGUF), Maya ships a **separate engine per model family** — `maya` for
GLM-5.3, `maya-ds4` for DeepSeek-V4.1 (with the pinned official prompt
encoder, numeric reasoning budgets, ID-matched parallel tool results, and
Engram n-gram tables), `maya-mm3` for MiniMax-M3.1 (contributed by
@needmorevram). The engine knows the model's prompt format, thinking
levels, MTP block, and vision tower natively.

### 2. Three-Tier Expert Placement with Auto-Calibration

Hot experts live in VRAM, warm in host RAM, cold are read in place from
NVMe. `./maya.sh --calibrate` tunes the placement to the machine in
10–15 min; setup recommends the context length (256K when VRAM+RAM hold
the model, 64K otherwise). Comparable in spirit to FreeToken's `q*`
bandwidth-adaptive split, but Maya's approach is placement/caching rather
than per-step PCIe-vs-CPU co-execution — and it goes one tier deeper
(SSD) by design.

### 3. Own Quant Tiers with Published Accuracy

Maya publishes its own GGUF quants per model, tiered by expert bit-width:

| Tier | Example size (GLM-5.3) | Measured vs FP8 original |
|------|----------------------|--------------------------|
| Maya-XS | ~94–96 GB | ~1.6-bit experts; fits a 128 GB box's memory |
| Maya-S / S24 | 96.5 / 94.7 GB | 97.7–97.9% zero-shot; 83% same next token |
| Maya-M | 116 GB | 97.9%; 86% — calibrated toward tool calls/front-end code |
| Maya-L | 156.3 GB | **99.2%**; 90% |

Each keeps the model's MTP block (drafts tokens ahead). Community quants
(EXL3, unsloth GGUF, abliterated variants) are also selectable via setup
ids. No other runner in the tier map publishes quant-quality numbers this
way — vLLM/SGLang/FreeToken serve whatever checkpoint you hand them.

### 4. Everything-in-the-Box UX

- **Browser dashboard**: chat (search, pins, folders, per-chat
  instructions/thinking, edit/regenerate) + live hardware monitor down to
  per-GPU fill, plus a "Saved" line computing what the model's own API
  would have charged.
- **APIs**: OpenAI-compatible `/v1`, Anthropic `/v1/messages` (Claude
  Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`), plus `/v1/responses`
  with schema-validated output and background tasks. Streamed tool calls.
- **Parallel requests**: `--parallel` up to 8, sharing dense-weight reads
  (×1.4–2.0 aggregate throughput in their bench data).
- **KV streaming**: `--kv-streaming on` keeps attention KV in RAM with a
  VRAM window — 1M context on a single RTX 3090.
- **DFlash2 drafter**: optional, up to 7 tokens/round (+44% on 9 GPUs);
  CC BY-NC-ND, opt-in only.
- **DS4 conversation slots** (DeepSeek engine): parked conversations keep
  state in RAM or spill to a `slots/` folder across restarts —
  a coarse version of session persistence for agent parent/worker splits.

## Relevance to the DGX Spark Setup

### Where Maya Fits in the Current ADR

| Scenario | Current Default | Maya Impact |
|----------|----------------|-------------|
| In-VRAM dense, max throughput | SGLang (cu130) | ❌ No impact — Maya is MoE-only, 3 families |
| In-VRAM MoE, multi-agent long-context | FreeToken | ➖ Partial contender — only for its 3 models; no semantic anchor caching |
| GPTQ / compressed-tensors / multi-GPU PP | vLLM | ❌ No impact |
| Brand-new arch / GGUF-only | llama.cpp | ❌ No impact (Maya *consumes* GGUF but only its own builds) |
| Frontier MoE not fitting VRAM | **FreeToken** (hybrid); Colibri emergency | ⚔️ **Direct contender — for its 3 supported models only** |
| Code execution | Glom | ❌ No impact |

### Why Maya Is a Serious Contender (for its three models)

1. **Purpose-built beats general-purpose — probably.** A per-model engine
   with model-native prompt format, MTP/drafter integration, and
   tuned quant tiers should beat a generic hybrid offload path on the
   same hardware. Their real-user table backs this: 124 tok/s on 2×H100,
   118 on 4×4090, 81 on RTX PRO 6000, 30+ on a single 5090 — for a
   321B model. Needs verification on GB10.
2. **GB10 data point exists.** 16.5 tok/s decode / ~730 tok/s prefill on
   DGX Spark (GB10, 121 GB unified) running Maya-S at 128K context — the
   only runner in this comparison with a published Spark number.
   FreeToken's SM103 support is still *unverified* (open question in the
   amendment ADR).
3. **Anthropic API + tool calls out of the box.** Matches the Claude
   Code / multi-agent provider surface without an adapter.
4. **The quant work is done.** Maya's XS/S/M/L tiers with measured
   accuracy deltas remove the re-quantization burden that FreeToken
   imposes on GPTQ models (see the FreeToken note's Option D).
5. **MIT, single maintainer + co-maintainer, extremely active**
   (pushed today, 87 open issues — a fast-moving tracker).

### Why FreeToken Stays the General MoE Default

1. **Coverage**: FreeToken serves 20+ MoE checkpoints (Qwen3.5/3.6-A3B,
   gpt-oss, Gemma-4, Muse-Glimmer, GLM-4.7/5.2…). Maya serves 3 families.
   The repo's model rotation depends on checkpoint breadth.
2. **Semantic-aware agentic caching**: anchor checkpoints for mid-session
   context edits are purpose-built for the 32–128 agent workload. Maya's
   DS4 conversation slots are parking/swap, not anchor-point reuse, and
   DeepSeek-engine only.
3. **Elastic memory**: FreeToken re-allocates VRAM between expert cache
   and KV at runtime without restart; Maya pins nearly all free RAM at
   startup (see below).
4. **Concurrency ceiling**: Maya's `--parallel` tops out at 8; the repo
   targets 32–128 concurrent agents. Maya is a *few-session* engine,
   FreeToken is a *fleet* engine.

### Deployment Frictions on the GB10 Box

- **RAM pinning**: the engine "pins most of the free RAM (all but about
  6 GB)" for its expert tier. On the Spark box running several `model-*.sh`
  servers simultaneously, Maya effectively wants the machine to itself —
  co-location with SGLang/vLLM/FreeToken instances on other ports needs
  verification or `--vram-reserve`-style limits.
- **Build toolchain**: `./setup.sh` compiles CUDA locally — needs nvcc +
  NVTX/profiler headers (CUDA 12.8+ for Blackwell/SM103; the README's
  troubleshooting documents the `nvidia-cuda-profiler-api`/`nvidia-nvtx`
  header wheels for CUDA 13). Not a Docker image like SGLang/vLLM.
- **Port convention**: Maya defaults to 8080; the repo reserves
  8000–8003 for SGLang/vLLM and 8004+ for FreeToken. Run Maya on 8006+
  (`./maya.sh --setup --port 8006`) to stay in-convention.
- **Unified memory nuance**: on GB10's 121 GB unified pool, the
  GLM-5.3 Maya-S (96.5 GB) and MiniMax XS (93 GB) tiers fit almost
  entirely in "VRAM"; DeepSeek Maya-XS (277 GB) still streams from SSD.
  The S24 tier (4-bit attention, tuned for ≤24 GB cards) is probably the
  wrong pick on unified memory — S or M is the better starting point.

## Proposed ADR Update

Amend the frontier-MoE row of the scenario map:

| Scenario | Default | Fallback / alternative |
|----------|---------|------------------------|
| Frontier MoE that does not fit in VRAM | **FreeToken** (hybrid) — general default | **Project Maya** — preferred for GLM-5.3-Flash / DeepSeek-V4.1 / MiniMax-M3.1 *if* GB10 benchmark ≥ FreeToken; Colibri — emergency fallback; Remote API — temporary (violates self-hosted) |

And in the per-model-type table, mark Frontier MoE as "FreeToken default;
Maya preferred for its 3 supported families pending benchmark."

The formal amendment has been authored:
[[adr-20261027-project-maya-model-specific-engine]] — proposes Maya as
the preferred engine for its three supported families pending a GB10
head-to-head benchmark, with FreeToken retained as the general MoE
default.

## Quick Start on DGX Spark

```bash
git clone https://github.com/mw00/project-maya.git && cd project-maya

# Interactive setup — measures GPU/CPU/RAM/SSD, picks quant tier
./setup.sh                       # pick GLM-5.3 · Maya-S on GB10 (96.5 GB fits unified memory)

# Non-interactive equivalent
./setup.sh --model glm-maya-s --port 8006

# Tune to hardware (10-15 min, one-time per config)
./maya.sh --calibrate

# Smoke-test
curl http://127.0.0.1:8006/v1/models
curl http://127.0.0.1:8006/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3-flash", "messages": [{"role":"user","content":"ping"}], "max_tokens": 64}'

# Claude Code against the Anthropic endpoint
ANTHROPIC_BASE_URL=http://127.0.0.1:8006 claude

# Bench + report for the head-to-head vs FreeToken
./maya.sh --bench && ./maya.sh --report
```

## Open Questions

- [ ] Does Maya build and run cleanly on GB10 (SM103, CUDA 12.8+/13)?
      A user report exists (16.5 tok/s, #282) but not on this box — and
      FreeToken's SM103 support is equally unverified. First one
      confirmed working wins the slot by default.
- [ ] Head-to-head on GB10: Maya (`glm-maya-s`, 128K ctx) vs FreeToken
      `hybrid` on GLM-5.3-class MoE — decode tok/s, prefill on long
      prompts, and *multi-agent* throughput. Maya's ×8 parallel cap vs
      FreeToken's agentic caching is the crux.
- [ ] RAM pinning: can Maya co-exist with the repo's other resident
      servers, or does it need exclusive use of the box? Does
      `--vram-reserve-mib` or equivalent bound the expert tier?
- [ ] Does the Anthropic endpoint hold up under Claude Code's actual
      traffic (long prompts, tool schemas, prompt-progress streaming)?
- [ ] DS4 conversation slots vs FreeToken anchor checkpoints for
      agent parent/worker splits — is slot parking enough?
- [ ] Project longevity: 3 weeks old, single lead maintainer
      (bus-factor vs FreeToken's Zaharia/Stoica/Han/Keutzer team).
      Track stars/issue velocity before depending on it.
- [ ] Maya-M-Derisked (refusal-reduced weight edit) — worth evaluating
      for the agent fleet's refusal-rate problem?

## References

- Repo + bench table + docs: <https://github.com/mw00/project-maya>
- Maya quants: <https://huggingface.co/peasantsmith>
- Settings reference: `docs/SETTINGS.md`, `docs/SETUP.md`,
  `docs/AMD_MAYA.md`, `docs/WINDOWS.md`, `docs/RESPONSES.md` (in repo)
- Prior art this builds on: [Strata](https://github.com/Niko1221/Strata),
  [ggml/llama.cpp](https://github.com/ggml-org/llama.cpp),
  [exllamav3](https://github.com/turboderp-org/exllamav3),
  [beellama.cpp](https://github.com/Anbeeld/beellama.cpp)
- Current ADR: [[adr-202608021744-sglang-glom-runtime-mapping]]
- FreeToken comparison note: [[note-freetoken-edge-native-moe-serving]]
- Vault-side landscape matrix:
  `2ndbrain/.../AI Home Network/project-comparison-local-llm-model-runners-20261027-115706.md`
