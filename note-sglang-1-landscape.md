---
title: "SGLang 1.0: Stability Release, Bundled Diffusion, and Paged Experts — Landscape Re-evaluation"
date:
  created: 2026-11-24
  knowledge-basis: 2026-11-24
  last-used: 2026-11-24
tags:
  - research
  - inference-engine
  - sglang
  - moe
  - diffusion
  - adr-update-candidate
aliases:
  - SGLang 1.0 note
  - SGLang v1.0 landscape
related:
  - "[[adr-202608021744-sglang-glom-runtime-mapping]]"
  - "[[adr-20260826-freetoken-runtime-amendment]]"
  - "[[note-freetoken-edge-native-moe-serving]]"
  - "[[note-project-maya-frontier-moe-serving]]"
  - "[[runner-lib]]"
source:
  - "https://github.com/sgl-project/sglang/issues/50045"
  - "https://github.com/sgl-project/sglang/releases/tag/v0.5.23"
  - "https://www.lmsys.org/blog/2026-11-17-sglang-summit-recap/"
---

# SGLang 1.0: Stability Release, Bundled Diffusion, and Paged Experts

> Research note — evaluates what the SGLang 1.0 announcement + v0.5.23 change
> for the tiered runtime model in
> [[adr-202608021744-sglang-glom-runtime-mapping]].

## TL;DR

SGLang announced **v1.0** at SGLang Summit on 2026-11-12 (release tracker
[sgl-project/sglang#50045](https://github.com/sgl-project/sglang/issues/50045)).
It is a **stability release**, not a feature merger — but combined with
v0.5.23 (2026-11-02) it shrinks two gaps that previously pushed workloads to
other runners:

1. **Paged Experts** (opt-in) runs MoE models larger than GPU memory on a
   single consumer GPU, ~30x faster than CPU offload — contests FreeToken's
   frontier-MoE slot.
2. **SGLang Diffusion** ships in the same package and CLI — image, video,
   3D, world, and action pipelines with an OpenAI-compatible image/video
   API, Python SDK, and ComfyUI plugin — contests the standalone ComfyUI
   stack for plain generation.

The other runner slots are **unaffected**: no GGUF (Strata), no Apple
Silicon/Metal (MLX), no GPTQ-maturity advantage (vLLM), no aarch64 GGUF
tuning (Strata sm_121).

**Status as of 2026-11-24:** v1.0.0 is not yet tagged. The release tracker
is in progress; latest stable is **v0.5.23**, which this repo now pins
(`lmsysorg/sglang:v0.5.23-cu130`, arm64 build exists).

## What 1.0 is

Per the tracker, 1.0 hardens existing functionality: replacing silent
configuration changes with errors that name the setting, and defining a
compatibility policy for the 1.x series (CLI flags, env vars, HTTP routes +
fields, and the Python engine API stay stable across 1.x). Core features
(scheduling, KV cache, parallelism, PD disaggregation, speculative
decoding, server APIs) get bug fixes and tests, no restructuring.

Capability inventory 1.0 stabilizes:

- **Performance**: overlap scheduling, multi-stream execution, CUDA graphs
  (breakable + full-graph), Rust radix tree/router/frontend, prefix
  sharing, speculative decoding.
- **Production serving**: aggregated + PD-disaggregated serving, TP/DP/EP/
  PP/DCP/PCP/SP parallelism, HiCache + Linker + HiSparse, unified memory.
- **Multimodal generation**: SGLang Diffusion in the same package/CLI,
  composable CFG/TP/SP/DP parallelism, request-level quality tiers,
  component residency + layerwise offload, OpenAI-compatible image/video
  API, Python SDK, ComfyUI plugin.
- **Platforms**: NVIDIA (Hopper, Blackwell, Grace Blackwell, Vera Rubin),
  AMD Instinct, Intel Xeon + XPU, Ascend NPU, PPU, MUSA; TPU via
  SGLang-TorchTPU (added in v0.5.23).

v0.5.23 additionally shipped: Paged Experts (opt-in), megakernel (NVIDIA)
and monokernel (AMD) plugins for fused persistent decode (~44% faster
single-user decode on GB300, 75% on MI350X), LoRA adapters on quantized
MoE (FP8/NVFP4), HiCache file-based L3 for MLA decode-context-parallel
prefix cache, and MoE TTFT improvements (30-58% sooner first token on 8K
prompts on H200).

## Upgrade notes (0.5.x → 1.0)

| Change | Impact on this repo |
|--------|---------------------|
| `sgl-model-gateway` removed; `sgl-router` replaces it and ships in the Docker images | None — scripts call `sglang.launch_server` directly |
| Unsupported configs rejected at startup or with 4xx (no silent downgrade) | Startup may fail where flags were auto-adjusted — treat as fix-ups |
| Deprecated flags removed (`--disable-cuda-graph`, `--enable-dp-attention`, other 0.5.x deprecations) | None — not used in `runner-lib.sh` or any model script |
| Admin routes require `--admin-api-key` (weight updates, cache, LoRA, profiling); `--api-key` no longer unlocks them | Only if the operator uses those routes; the OpenAI serving API is unaffected |
| Media URLs resolving to private/loopback/link-local → 400, unless `--allow-private-media-hosts` | Relevant only for diffusion requests referencing tailnet/LAN media |

## Impact on the tiered runtime model

| ADR slot | Impact |
|----------|--------|
| SGLang — in-VRAM dense default | Unchanged, stronger (stability policy, megakernel opt-in) |
| FreeToken — MoE default (in-VRAM, agentic) | Unchanged — semantic anchor KV for agent loops is still unique |
| FreeToken — frontier MoE > VRAM | **Contested** — Paged Experts covers the same slot; validate on GB10 before amending the ADR |
| vLLM — GPTQ/compressed-tensors/multi-GPU PP | Unchanged |
| Strata — GGUF packs on GB10 | Unchanged — SGLang has no GGUF path |
| MLX — Apple Silicon | Unchanged — SGLang has no Metal/macOS support |
| ComfyUI (image/video gen, `comfyui/`) | **Contested long-term** — SGLang Diffusion serves generation via OpenAI API + ComfyUI plugin; the LoRA training + finishing pipeline has no SGLang equivalent yet |

## Action items

- [x] Bump `SGLANG_IMAGE` to `lmsysorg/sglang:v0.5.23-cu130` (arm64 present).
- [ ] Validate Paged Experts on the GB10 with a >128 GB MoE before
      considering an ADR amendment for the frontier-MoE slot.
- [ ] When v1.0.0 tags: re-pin, re-validate `awq_merlin` +
      `--mem-fraction-static` on GB10, run the 1.0 checklist in
      `internal-docs/guides/runtime-selection.md`.
- [ ] Evaluate SGLang Diffusion vs the `comfyui/` pipeline when a 1.x image
      is running on the host (plain generation only — keep ComfyUI for
      LoRA training + finishing).

## Sources

- SGLang 1.0 release tracker:
  <https://github.com/sgl-project/sglang/issues/50045>
- v0.5.23 release notes:
  <https://github.com/sgl-project/sglang/releases/tag/v0.5.23>
- SGLang Summit 2026 recap:
  <https://www.lmsys.org/blog/2026-11-17-sglang-summit-recap/>
- Vault feature matrix (2ndbrain):
  `Default/Technologies/Programming/Artificial Intelligence/Inference Runtimes/project-comparison-llm-inference-runtimes-20261124190326.md`
