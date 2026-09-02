---
modeline: "vim: set ft=markdown:"
title: "ADR: Add Mac Studio M5 Ultra 512GB as Second Inference Host — Independent Model Deployment, Not Disaggregated Prefill/Decode"
gdr-id: "gdr20260831001"
slug: "mac-studio-multi-host-deployment"
url: "https://github.com/levonk/levonk-ai-playground/blob/main/internal-docs/adr/2026/08/adr-20260831-mac-studio-multi-host-deployment.md"
synopsis: "When the Mac Studio M5 Ultra 512GB arrives, deploy it as a second independent single-machine inference host running different models from the DGX Spark — not as a prefill/decode split partner via EXO or similar frameworks. The Mac becomes the primary host for large and MoE models that exceed the DGX Spark's 128GB VRAM; the DGX Spark remains the primary host for models that fit in 128GB and benefit from CUDA compute density. This preserves the repo's single-machine-per-script architecture, respects the OOS 'not a multi-host orchestrator' boundary, and is strictly better than EXO-style disaggregation for the operator's stated goals (run multiple models simultaneously, run huge models). MoE models strengthen the case for independent deployment: sparse expert activation makes EXO's prefill/decode split less effective (reduced compute intensity during prefill), the Mac's 512GB unified memory eliminates the disk-streaming workaround needed for frontier MoE on the DGX Spark, and EXO is physically impossible for frontier MoE (GLM-5.2 753B) because the DGX Spark cannot hold the full weight copy that disaggregated prefill/decode requires. slotstream (carloslfu/slotstream) — a Swift MoE disk-streaming engine for Apple Silicon — is evaluated as a Mac-side fallback for models exceeding 512GB unified memory, but does not change the decision for the 512GB Mac Studio where models fit in unified memory; its measured 33GB expert-cache ceiling (beyond which SSD bandwidth bottlenecks decode at ~12 tok/s) reinforces the ADR's argument that unified memory breaks the disk-streaming performance class ceiling."
author: "https://github.com/levonk"
date-created: "2026-08-31"
date-updated: "2026-08-31"
date-review: "2027-02-28"
date-triggers: ["2026-11-30", "Mac Studio M5 Ultra release announcement"]
version: "0.1.0"
status: "proposed"
aliases: []
tags: [doc/architecture/gdr]
supersedes: []
superseded-by: []
related-to: ["adr-202608021744-sglang-glom-runtime-mapping", "adr-20260826-freetoken-runtime-amendment"]
scope:
  impact-scope:
    - "Repo scope: gains a second independent deployment target (Mac Studio), not a multi-host orchestration layer"
    - "model-*.sh runner scripts: new model-mac-*.sh variant or HOST/target parameter for Mac-bound models"
    - "runner-lib.sh / vllm-runner-lib.sh: add Mac-side launch path (MLX / llama.cpp / FreeToken-on-ARM, not Docker+vLLM)"
    - "README model table: add 'Host' column (DGX Spark vs Mac Studio) alongside existing port/runtime columns"
    - "Host port allocation: Mac Studio uses its own port namespace (independent host, ports can overlap with DGX Spark)"
    - "Frontier MoE serving: GLM-5.2 753B moves from DGX Spark (FreeToken hybrid/disk-stream) to Mac Studio (FreeToken in-unified-memory or MLX)"
    - "test-query-local.sh: already supports -h flag for remote host; no change needed, just documentation"
    - "OOS doc: amend 'not a multi-host orchestrator' to clarify that having two independent single-machine hosts is in-scope; building a control plane/scheduler/load balancer remains out-of-scope"
  excluded-scope:
    - "Multi-host orchestration, scheduling, load balancing, or control plane (remains OOS)"
    - "EXO or similar disaggregated prefill/decode frameworks (rejected — see Decision)"
    - "Cross-host tensor parallelism or expert parallelism (rejected — see Alternatives)"
    - "DGX Spark hardware configuration or NVIDIA driver changes"
    - "Mac Studio OS configuration or Apple Silicon kernel-level tuning"
hardware:
  target-primary: "NVIDIA DGX Spark (Blackwell GB10 128GB VRAM, ~100 TFLOPs FP16, 273 GB/s VRAM bandwidth)"
  target-secondary: "Apple Mac Studio M5 Ultra 512GB unified memory (bandwidth TBD at release; M3 Ultra was 819 GB/s, M5 Ultra expected higher), ~26+ TFLOPs FP16 GPU (Apple-silicon GPU, not CUDA)"
  interconnect: "10 GbE (existing home network). No NVLink, no InfiniBand."
  dgx-spark-role: "Models that fit in 128GB VRAM and benefit from CUDA compute density (dense models, GPTQ models, in-VRAM MoE with FreeToken fused backend)"
  mac-studio-role: "Large/MoE models that exceed 128GB (frontier MoE in unified memory, large dense models at higher precision, long-context models needing large KV cache)"
---

# Decision Record: Add Mac Studio M5 Ultra 512GB as Second Inference Host — Independent Model Deployment, Not Disaggregated Prefill/Decode

**Filename:** `adr-20260831-mac-studio-multi-host-deployment.md`

- belongs in `internal-docs/adr/2026/08/`
- relates to [ADR-202608021744](adr-202608021744-sglang-glom-runtime-mapping.md) (tiered runtime selection)
- relates to [ADR-20260826-freetoken-runtime-amendment](adr-20260826-freetoken-runtime-amendment.md) (FreeToken for MoE)

---

## Context

The operator currently runs a single NVIDIA DGX Spark (GB10, 128GB VRAM) and is
memory-constrained — one model at a time fills the box, and frontier MoE models
(GLM-5.2 753B) require disk-streaming workarounds (FreeToken hybrid backend or
Colibri). The Mac Studio M5 Ultra with 512GB unified memory is expected to ship
in ~2 months (as of 2026-08-31). The operator's goals are:

1. **Run multiple models simultaneously** (not possible today — one model fills
   the DGX Spark).
2. **Run huge models** (frontier MoE, large dense models at higher precision)
   that don't fit in 128GB.

The question is how to use the two machines together. Three architectures were
considered:

- **Option A — Independent deployment**: Different models on each machine. Each
  machine runs its own model(s) independently, exposed on its own host/port.
- **Option B — Disaggregated prefill/decode (EXO-style)**: Both machines
  collaborate on a single model, with the DGX Spark handling prefill (compute-
  bound) and the Mac Studio handling decode (memory-bandwidth-bound), connected
  via EXO or a similar framework. KV cache is streamed layer-by-layer from the
  prefill machine to the decode machine.
- **Option C — Cross-host weight sharding (tensor/expert parallelism)**: Split
  the model's weights across both machines, with each machine computing its
  shard and synchronizing via the network.

### Reference: EXO Labs blog post

The operator referenced the EXO Labs blog post
([blog.exolabs.net/nvidia-dgx-spark](https://blog.exolabs.net/nvidia-dgx-spark/))
which benchmarks a DGX Spark + Mac Studio M3 Ultra combination using EXO 1.0 for
disaggregated prefill/decode. Their result: Llama-3.1 8B (FP16) with 8k context
achieves 2.8x speedup vs Mac Studio alone, by using the DGX Spark's 4x compute
advantage for prefill and the Mac Studio's 3x bandwidth advantage for decode.

**Critical observation**: EXO's benchmark uses Llama-3.1 8B — a model that fits
trivially in both machines. The speedup is purely latency, not capacity. The
blog does not address running multiple models or running models larger than the
smaller machine's memory.

### MoE considerations

The operator's workload is MoE-heavy (Qwen3-Next-80B-A3B, Qwen3-Coder-Next,
GLM-5.2 753B). MoE models change the prefill/decode analysis in ways that the
EXO blog (which benchmarks a dense 8B model) does not cover. These are analyzed
in detail in the Rationale section.

## Decision

**Deploy the Mac Studio M5 Ultra as a second independent single-machine
inference host. Run different models on each machine. Do NOT use EXO-style
disaggregated prefill/decode or cross-host weight sharding.**

### Role assignment

| Machine | Primary role | Models | Runtime |
|----------|-------------|--------|---------|
| **Mac Studio M5 Ultra 512GB** | Large/MoE models, frontier MoE, high-precision dense | GLM-5.2 753B (NVFP4), large dense models at BF16/FP8, long-context models needing large KV cache | MLX (Apple-native), FreeToken-on-ARM (if supported), or llama.cpp |
| **DGX Spark 128GB** | Models that fit in 128GB VRAM and benefit from CUDA compute | Qwen3-Next-80B-A3B-AWQ-4bit, Qwen3-Coder-Next-AWQ-4bit, reasoning distillations, any GPTQ model | SGLang (dense), FreeToken (in-VRAM MoE), vLLM (GPTQ fallback), llama.cpp (new-arch) |

### What this means concretely

1. **The repo gains a second deployment target, not a multi-host orchestration
   layer.** Each `model-*.sh` script (or a new `model-mac-*.sh` variant) targets
   one machine. The operator runs scripts on each machine independently. No
   control plane, no scheduler, no load balancer — consistent with the OOS doc.

2. **The Mac Studio runs Apple-silicon-native runtimes, not Docker+vLLM.** vLLM
   requires CUDA and does not run on Apple Silicon. The Mac-side runtime is one
   of: MLX (Apple's native ML framework, best Apple-silicon optimization),
   llama.cpp (GGUF, broad model support), or FreeToken if it ships an ARM/Apple-
   Silicon backend. This is a new runtime path in the runner library, parallel
   to the existing SGLang/vLLM/FreeToken paths on the DGX Spark.

3. **Frontier MoE (GLM-5.2 753B) moves to the Mac Studio.** At NVFP4 (~380GB),
   it fits in the Mac's 512GB unified memory and runs from fast memory instead
   of disk. This eliminates the FreeToken hybrid / Colibri disk-streaming
   workaround currently required on the DGX Spark. The DGX Spark's 128GB cannot
   hold the full weight copy, so EXO disaggregation is physically impossible for
   this model (see Rationale).

4. **`test-query-local.sh` already supports remote hosts** via `-h` — no script
   change needed, just documentation showing the Mac Studio as a target.

5. **Host ports can overlap between machines.** Since each machine is
   independent, the Mac Studio can use port 8000 for its model while the DGX
   Spark uses 8000 for its model. The operator addresses them by host
   (`ai-dgx:8000` vs `ai-mac:8000`), not by port alone.

### What does NOT change

- The repo remains a single-machine-per-script playground. No orchestration
  layer is added.
- The DGX Spark's existing runtime selection (ADR-202608021744) is unchanged.
  SGLang, vLLM, FreeToken, llama.cpp, and Glom keep their roles on the DGX Spark.
- The OOS boundary ("not a multi-host orchestrator") is respected — we are adding
  a second independent host, not building a distributed system. The OOS doc
  should be amended to clarify this distinction (see Rollout).
- The `runner-lib.sh` / `vllm-runner-lib.sh` dispatcher pattern is preserved. A
  new `run_mlx()` or `run_llamacpp_mac()` function is added alongside the
  existing functions, not replacing them.

## Rationale

### Why independent deployment, not EXO disaggregation

The EXO blog demonstrates a real speedup (2.8x) for a specific scenario: a
single model that fits in both machines, with large context, where latency is
the only goal. The operator's goals are different:

| Operator goal | EXO disaggregation | Independent deployment |
|---------------|:------------------:|:----------------------:|
| Run multiple models simultaneously | ❌ Both machines tied to one model | ✅ Each machine runs its own model(s) |
| Run huge models (>128GB) | ❌ Both machines need full weight copy; capped at smaller machine's memory | ✅ Mac's 512GB alone fits models ~4x larger |
| Memory efficiency | ❌ Weights duplicated on both machines | ✅ Each weight loaded once, on one machine |
| Failure isolation | ❌ Either machine down = model down | ✅ Independent failure domains |
| Latency on a single fit-able model | ✅ ~2.8x faster (large context) | Neutral |
| Operational complexity | ❌ EXO mesh, profiling, KV streaming | ✅ Just run two scripts |

The EXO split is a **latency technique, not a capacity technique**. Both
machines must load the full model weights — the only thing that crosses the
network is the KV cache. You cannot run a model larger than your smallest
participating machine's free memory this way. For the operator's goals (multiple
models, huge models), independent deployment is strictly better.

### Why MoE models strengthen the case against EXO disaggregation

The EXO blog benchmarks Llama-3.1 8B, a **dense** model. MoE models change the
prefill/decode analysis in three ways that make EXO's split less attractive:

#### 1. Sparse activation reduces prefill compute intensity

EXO's speedup depends on overlapping KV cache transfer with prefill compute.
The overlap works when prefill compute time (tcomp) exceeds KV transfer time
(tsend). For dense models, prefill compute is Θ(s²) per attention layer plus
Θ(s) per FFN layer — the FFN contributes significant compute.

For MoE models, the FFN is **sparse**: only the active experts (e.g., 3B of 80B
parameters for Qwen3-Next-80B-A3B) are computed per token. This reduces the
FFN compute by the sparsity ratio (e.g., ~27x less for 80B-A3B). The attention
compute (Θ(s²)) is unchanged, but the total per-layer compute F is smaller,
making tcomp smaller, making the overlap harder to achieve.

Concretely: EXO's threshold for full overlap is `s > P/B · q/K`, where K is the
attention architecture constant (K=16 for GQA models like Llama-3 70B). MoE
sparsity doesn't change K (which is about attention heads), but it reduces the
effective compute per layer, which means the actual overlap threshold is higher
than the blog's formula predicts for MoE models. The DGX Spark's prefill
advantage is diminished for MoE because there's less compute to be fast at.

#### 2. Frontier MoE makes EXO physically impossible

GLM-5.2 753B at NVFP4 is ~380GB of weights. EXO disaggregation requires both
machines to hold the full model:

- **Mac Studio (512GB)**: Can hold 380GB of weights. ✅
- **DGX Spark (128GB)**: Cannot hold 380GB of weights. ❌

The DGX Spark would need to disk-stream its weight copy during prefill, which
defeats the purpose — prefill is supposed to be compute-bound on the fast GPU,
not I/O-bound on disk. EXO's prefill/decode split is **physically impossible**
for frontier MoE on this hardware pair.

Independent deployment solves this directly: the Mac Studio holds the entire
model in 512GB unified memory and serves it alone. No disk streaming, no
network transfer, no second machine needed.

#### 3. MoE memory capacity is the real bottleneck, and the Mac solves it directly

MoE models have large total parameter counts (all experts must be stored) but
sparse activation (only a few experts compute per token). The bottleneck for
MoE serving is **memory capacity** (storing all experts), not compute. This is
the opposite of dense models, where compute and bandwidth are the bottlenecks
and capacity is usually sufficient.

The Mac Studio's 512GB unified memory at 819+ GB/s (M3 Ultra baseline; M5 Ultra
expected higher) directly addresses the MoE memory capacity bottleneck:

- **On the DGX Spark (128GB)**: Frontier MoE requires FreeToken's `hybrid`
  backend or Colibri's disk-streaming — experts are fetched from disk or host
  RAM on demand, with tokens/sec limited by disk/NVMe bandwidth (~3-7 GB/s for
  NVMe, much less for SSD).
- **On the Mac Studio (512GB unified memory)**: The entire MoE model fits in
  unified memory. Active experts are served from 819+ GB/s unified memory
  instead of ~3-7 GB/s NVMe. This is a **100-250x bandwidth improvement** for
  expert fetching, which is the dominant cost in MoE inference.

The Mac Studio doesn't just "have more memory" — for MoE models, its unified
memory architecture is the ideal fit for the sparse-expert access pattern. The
GPU and CPU share the same memory pool, so expert activation doesn't require a
PCIe copy from host RAM to VRAM (as it would on the DGX Spark's hybrid backend).
The expert is already in the GPU's address space.

### Why not cross-host weight sharding (tensor/expert parallelism)

**Option C** — splitting model weights across both machines — was considered
and rejected:

1. **Heterogeneous architecture**: The DGX Spark is NVIDIA/CUDA (SM103,
   Blackwell); the Mac Studio is Apple Silicon (Metal, ARM). No single
   inference framework supports tensor parallelism across CUDA and Metal. You
   would need a custom shim translating between NVIDIA collective ops and Apple
   Metal ops — this does not exist.

2. **Network bottleneck**: Tensor parallelism requires all-reduce after every
   transformer layer; expert parallelism requires all-to-all communication after
   the MoE routing. Over 10 GbE (~1.25 GB/s practical), the per-layer
   synchronization cost would dominate. NVLink is ~300 GB/s; 10 GbE is ~240x
   slower. The computation would never amortize the communication.

3. **No framework support**: EXO does not do cross-host weight sharding today.
   vLLM's tensor parallelism requires homogeneous GPUs. SGLang's distributed
   inference requires same-architecture nodes. FreeToken's hybrid backend is
   single-host (VRAM + host RAM on one machine). Building this from scratch is
   not feasible for a single-operator playground.

4. **OOS violation**: Cross-host weight sharding is a distributed inference
   system — exactly the kind of multi-host orchestration the OOS doc excludes.

### Why the Mac Studio is the right home for large/MoE models (not the DGX Spark)

| Dimension | DGX Spark (128GB) | Mac Studio M5 Ultra (512GB) |
|-----------|-------------------|------------------------------|
| Memory capacity | 128GB VRAM | 512GB unified memory |
| Memory bandwidth | 273 GB/s (VRAM) | 819+ GB/s (unified, M3 Ultra baseline) |
| Compute | ~100 TFLOPs FP16 (CUDA) | ~26+ TFLOPs FP16 (Apple GPU) |
| MoE expert access | VRAM (if fits) or NVMe disk (~3-7 GB/s) | Unified memory (819+ GB/s) — no copy needed |
| Frontier MoE (GLM-5.2 753B NVFP4 ~380GB) | Disk-streaming required (FreeToken hybrid / Colibri) | Fits entirely in unified memory |
| Dense model compute | ✅ Fast (CUDA, 100 TFLOPs) | Slower (Apple GPU, ~26 TFLOPs) |
| vLLM / SGLang support | ✅ Full CUDA support | ❌ No CUDA — requires MLX / llama.cpp |
| GPTQ support | ✅ vLLM | ❌ Not on Apple Silicon |

The Mac Studio is worse at compute (dense model prefill) but dramatically better
at memory capacity and bandwidth (MoE expert access, large model storage). The
DGX Spark is better at compute but memory-constrained. Independent deployment
lets each machine play to its strength.

## MoE-Specific Analysis

### How MoE models interact with each deployment option

| Option | Dense models | MoE models (fit in 128GB) | Frontier MoE (>128GB) |
|--------|:------------:|:-------------------------:|:---------------------:|
| **A: Independent** | Each machine runs its own dense model. DGX Spark is faster for compute-bound dense. | Each machine runs its own MoE model. DGX Spark uses FreeToken fused (in-VRAM); Mac uses MLX/llama.cpp from unified memory. | Mac Studio holds full model in unified memory. DGX Spark cannot participate. ✅ |
| **B: EXO split** | ✅ 2.8x speedup (blog-validated). Both machines hold full weights. | ⚠️ Reduced speedup — sparse activation lowers prefill compute intensity, making KV overlap harder. Both machines hold full weights. | ❌ Physically impossible — DGX Spark (128GB) cannot hold 380GB weight copy. |
| **C: Weight sharding** | ❌ 10 GbE too slow for per-layer all-reduce. Heterogeneous arch. | ❌ 10 GbE too slow for all-to-all expert routing. Heterogeneous arch. No framework support. | ❌ Same as above. |

### MoE sparse activation and the EXO overlap threshold

EXO's layer-by-layer KV streaming hides communication when `tsend < tcomp`,
i.e., when the KV transfer time for a layer is less than the compute time for
the next layer. For dense models:

```
tcomp = (F_attention + F_ffn) / P
```

where `F_attention ~ c1·s²` (quadratic in sequence length) and `F_ffn ~ c2·s`
(linear, full FFN). The FFN contributes meaningfully to tcomp.

For MoE models:

```
tcomp = (F_attention + F_ffn_sparse) / P
```

where `F_ffn_sparse = (active_params / total_params) · F_ffn_dense`. For
Qwen3-Next-80B-A3B, the sparsity ratio is 3B/80B = 3.75%, so `F_ffn_sparse` is
~27x smaller than `F_ffn_dense`. This reduces tcomp by a factor that depends on
the attention-to-FFN compute ratio for the model, but the direction is clear:
**MoE prefill has less compute per layer, making the KV transfer overlap harder
to achieve.**

The EXO blog's threshold formula `s > P/B · q/K` assumes dense FFN compute. For
MoE, the effective threshold is higher (you need a longer context to achieve
full overlap), meaning the speedup is smaller at any given context length.

### MoE expert access patterns favor unified memory

In MoE inference, each token activates a small number of experts (e.g., 8 of
256 for GLM-5.2). The access pattern is:

1. Router selects experts for each token.
2. Selected expert weights must be in fast memory for the FFN computation.
3. Different tokens may select different experts, so many experts may be needed
   per batch.

On the DGX Spark with FreeToken hybrid backend:
- Experts that fit in VRAM are in VRAM (273 GB/s access).
- Experts that don't fit are streamed from host RAM or NVMe (~3-7 GB/s for
  NVMe, ~50-100 GB/s for host RAM over PCIe).
- The hybrid split is calibrated by `ft bench bw` to find the optimal
  VRAM-vs-host-RAM ratio.

On the Mac Studio with unified memory:
- ALL experts are in unified memory (819+ GB/s access).
- No VRAM-vs-host-RAM split needed — the GPU and CPU share the same memory
  pool.
- No PCIe bottleneck — expert weights are already in the GPU's address space.
- No disk streaming — everything is in RAM.

For GLM-5.2 753B (NVFP4, ~380GB):
- **DGX Spark**: 128GB VRAM holds ~34% of experts; the rest streams from host
  RAM / NVMe. Tokens/sec is bottlenecked by expert fetch bandwidth.
- **Mac Studio**: 512GB unified memory holds 100% of experts. Tokens/sec is
  bottlenecked by unified memory bandwidth (819+ GB/s), which is ~100-250x
  faster than NVMe streaming.

This is the single largest win from adding the Mac Studio: **frontier MoE
serving moves from disk-bound to memory-bandwidth-bound**, which is a
fundamental performance class change, not an incremental improvement.

### MoE on the Mac Studio: runtime options

| Runtime | MoE support | Apple Silicon | Loading model | Status |
|---------|:-----------:|:-------------:|:-------------:|--------|
| **MLX** | ✅ (Apple's native framework, optimized for unified memory) | ✅ Native | Full in-memory (all-or-nothing — see note below) | Production-ready for Apple Silicon |
| **llama.cpp** | ✅ (GGUF MoE support, Metal backend) | ✅ Metal | Full in-memory or mmap (llama.cpp can mmap partial tensors, unlike MLX) | Production-ready, broad model support |
| **slotstream** | ✅ (disk-streaming, `pread` expert cache) | ✅ Native Swift/Metal | Disk-stream from SSD into fixed expert cache pool | Working but single-model (Qwen3.8-Flash-Next only), no tool calls |
| **FreeToken** | ✅ (MoE-focused, but CUDA-centric) | ❓ Unknown | Full in-memory or hybrid (VRAM+host RAM) | Needs ARM/Apple-Silicon backend — investigate |
| **vLLM** | ✅ | ❌ No CUDA | — | Not available on Apple Silicon |
| **SGLang** | ✅ | ❌ No CUDA | — | Not available on Apple Silicon |

**MLX partial-materialization limitation**: MLX cannot materialize part of a
memory-mapped MoE tensor. `mlx_lm.load()` evaluates all experts of a layer
(e.g., all 512 for Qwen3.8-Flash-Next), loading the full ~100GB, and dies on
Macs with less memory than the model size. This is documented by the
[slotstream](https://github.com/carloslfu/slotstream) project, which built a
custom `pread`-based expert cache specifically to work around this MLX
limitation. **On the 512GB Mac Studio this is not a problem** — the full model
fits in unified memory and MLX loads it all. But it means MLX cannot serve a
model larger than available unified memory, whereas llama.cpp (mmap) and
slotstream (disk-streaming) can.

**slotstream — the Mac disk-streaming fallback**: slotstream is the Apple-
Silicon equivalent of FreeToken's hybrid backend or Colibri on the DGX Spark.
It streams MoE experts from SSD into a fixed cache pool via `pread`, allowing
a model larger than memory to run. Key measured findings from its README (on a
48GB M5 Pro, Qwen3.8-Flash-Next 125B 4-bit, 104GB on disk):

- ~12 tok/s warm decode, 32GB peak memory (auto-sized)
- **33GB expert cache ceiling**: beyond 33GB of expert cache, decode speed
  does not improve. A 128GB Mac gets the same plan as a 48GB one. The
  bottleneck is SSD fetch bandwidth (~3-7 GB/s), not cache pool size.
- Prefill is slow: 8,000 tokens = ~1 minute. Capped at 32k context.
- Single model only (v0 runs exactly `qwen3.8-flash-next:4bit`), no tool
  calls, no images, no JSON-schema output.

**slotstream does not change the ADR's decision for the 512GB Mac Studio.**
The models the operator wants to run (GLM-5.2 753B at ~380GB, Qwen3.8-Flash-
Next at 104GB) fit entirely in 512GB unified memory. slotstream's disk-
streaming is unnecessary — MLX or llama.cpp with full in-memory loading at
819+ GB/s unified memory bandwidth should be dramatically faster than
slotstream's ~12 tok/s (which is SSD-bandwidth-bound at ~3-7 GB/s).

**slotstream is relevant as a fallback** if:
1. A future model exceeds 512GB and MLX's all-or-nothing loading can't handle
   it (llama.cpp's mmap or slotstream's `pread` would be needed).
2. The operator uses a smaller Mac (48GB, 128GB) where the model doesn't fit.
3. MLX's MoE support for a specific model architecture is broken or missing
   and slotstream happens to support it.

**slotstream's 33GB ceiling finding strengthens the ADR's core argument**:
disk-streaming MoE has a hard speed ceiling set by SSD bandwidth, not cache
size. The Mac Studio's 512GB unified memory doesn't just add capacity — it
breaks through the disk I/O ceiling entirely. All experts are in 819+ GB/s
unified memory instead of behind a 3-7 GB/s SSD wall. This is the same
fundamental performance class change the ADR identifies for FreeToken hybrid
on the DGX Spark: moving from disk-bound to memory-bandwidth-bound.

**Recommendation**: Start with MLX for Apple-native MoE models (if NVFP4
checkpoints are available in MLX format) and llama.cpp for GGUF MoE models.
Keep slotstream as a documented fallback for models that exceed unified
memory or for smaller Macs. Investigate FreeToken Apple Silicon support as a
stretch goal — if FreeToken ships an ARM backend, its semantic-aware caching
and bandwidth-adaptive execution would be valuable on the Mac too.

## Affected Components

- **`runner-lib.sh` / `vllm-runner-lib.sh`** — add `run_mlx()` and/or
  `run_llamacpp_mac()` functions for Mac-side deployment. These do not use
  Docker (MLX and llama.cpp run natively on macOS). The function launches the
  server directly on the Mac host, similar to the FreeToken native Python path.
- **New `model-mac-*.sh` scripts** — one per model served on the Mac Studio.
  Initial set: `model-mac-frontier.sh` (GLM-5.2 753B NVFP4, MLX or llama.cpp,
  port 8000 on the Mac), `model-mac-large-dense.sh` (large dense model at
  higher precision, port 8001 on the Mac).
- **Existing `model-*.sh` scripts** — unchanged. They continue to target the
  DGX Spark. The operator runs DGX scripts on the DGX Spark and Mac scripts on
  the Mac Studio.
- **`test-query-local.sh`** — already supports `-h` for remote host. No code
  change. Update documentation to show Mac Studio as a target
  (`./test-query-local.sh -h ai-mac -p 8000`).
- **README model table** — add "Host" column (DGX Spark vs Mac Studio) alongside
  existing port/runtime columns. Add Mac Studio models.
- **ADR-202608021744** — add a note that the runtime selection applies to the
  DGX Spark; the Mac Studio has its own runtime selection (MLX / llama.cpp /
  FreeToken-on-ARM). Cross-reference this ADR.
- **ADR-20260826-freetoken-runtime-amendment** — add a note that FreeToken's
  frontier MoE role (hybrid backend for GLM-5.2) is superseded on the Mac
  Studio by in-unified-memory serving. FreeToken hybrid remains the DGX Spark
  fallback if the Mac Studio is unavailable.
- **OOS doc** — amend "not a multi-host orchestrator" to clarify: having two
  independent single-machine hosts is in-scope; building a control plane,
  scheduler, or load balancer across them remains out-of-scope.
- **AGENTS.md** — update hardware target section to mention the Mac Studio as
  a second target. Update the "Hardware target" universal contract.

## Consequences

### Positive

- **Multiple models run simultaneously.** The DGX Spark runs its model(s), the
  Mac Studio runs its model(s). The operator can query both at the same time.
  This directly solves the "feeling constrained running 1 model" problem.
- **Frontier MoE moves from disk-bound to memory-bandwidth-bound.** GLM-5.2
  753B on the Mac Studio's 512GB unified memory (819+ GB/s) is fundamentally
  faster than on the DGX Spark's hybrid backend (NVMe ~3-7 GB/s for the
  overflow experts). This is a performance class change, not an incremental
  improvement.
- **No memory waste.** Each model's weights are loaded once, on one machine.
  EXO disaggregation would duplicate weights on both machines.
- **Independent failure domains.** If the DGX Spark is down, the Mac Studio
  keeps serving. If the Mac Studio is down, the DGX Spark keeps serving.
- **No new distributed-system complexity.** No EXO mesh, no profiling, no KV
  streaming, no cross-host synchronization. Just two independent scripts.
- **The repo's architecture is preserved.** Single-machine-per-script, shell
  library, unique host port per model. The Mac Studio is just a second target
  for the same pattern.

### Negative

- **Two different runtime ecosystems to manage.** The DGX Spark uses
  SGLang/vLLM/FreeToken (CUDA). The Mac Studio uses MLX/llama.cpp (Metal). The
  operator must be familiar with both. The runner library gains Mac-specific
  functions alongside the existing CUDA functions.
- **No compute-bandwidth split for latency-sensitive single-model workloads.**
  If the operator later wants to optimize a single fit-able model for minimum
  latency (not capacity, not multi-model), EXO's 2.8x speedup is forfeited.
  This is the right trade-off for the current goals but worth noting.
- **Mac Studio M5 Ultra does not exist yet.** This ADR is proposed ahead of the
  hardware. The M5 Ultra's specs (memory bandwidth, TFLOPs, unified memory
  size) are assumed from the M3 Ultra baseline and Apple's trajectory. The ADR
  should be validated when the M5 Ultra ships.
- **MLX MoE support and performance are unverified for frontier MoE.** MLX is
  production-ready for Apple Silicon but its MoE support for GLM-5.2 753B
  specifically needs validation. llama.cpp's GGUF MoE support is broader but
  may be slower than MLX for Apple-native formats.
- **NVFP4 on Apple Silicon is unverified.** The DGX Spark uses NVFP4 (NVIDIA's
  4-bit format). The Mac Studio may need a different quantization (GGUF Q4_K_M,
  MLX 4-bit, or Apple's own format). Re-quantization may be needed for Mac-side
  models.

### Neutral

- **Host ports can overlap between machines.** This is a feature (independence)
  but means the operator must address by host, not port alone.
- **The Mac Studio does not run Docker.** MLX and llama.cpp run natively on
  macOS. This is consistent with the FreeToken native Python path already
  established in ADR-20260826.

## Rollout

1. **Wait for Mac Studio M5 Ultra release.** Validate actual specs: unified
   memory size (confirm 512GB), memory bandwidth (confirm ≥819 GB/s), GPU
   TFLOPs. If specs differ significantly from assumptions, revisit this ADR.
2. **Validate MLX MoE support.** Install MLX on the Mac Studio, load a small
   MoE model (e.g., Qwen3-Next-80B-A3B in MLX 4-bit format), and confirm it
   serves correctly. Benchmark tokens/sec and TTFT.
3. **Validate llama.cpp MoE on Apple Silicon.** Load the same MoE model in
   GGUF format via llama.cpp with the Metal backend. Compare performance vs
   MLX.
4. **Add `run_mlx()` and/or `run_llamacpp_mac()` to the runner library.**
   Follow the FreeToken native Python pattern (no Docker).
5. **Create `model-mac-frontier.sh`** (GLM-5.2 753B, MLX or llama.cpp, port
   8000 on the Mac). Smoke-test with
   `./test-query-local.sh -h ai-mac -p 8000`.
6. **Create `model-mac-large-dense.sh`** for large dense models that benefit
   from the Mac's 512GB memory (e.g., a 70B model at BF16 or FP8 instead of
   4-bit quantization).
7. **Update README model table** — add "Host" column and Mac Studio rows.
8. **Update ADR-202608021744** — add cross-reference to this ADR for the Mac
   Studio runtime selection.
9. **Update OOS doc** — clarify that two independent single-machine hosts are
   in-scope; multi-host orchestration remains out-of-scope.
10. **Investigate FreeToken on Apple Silicon** — if FreeToken ships an ARM/
    Metal backend, evaluate it as the Mac-side MoE runtime (semantic-aware
    caching would be valuable for agent loops on the Mac too).
11. **Investigate NVFP4 → MLX/GGUF re-quantization** — determine whether
    NVFP4 checkpoints can be converted to MLX or GGUF format for Mac-side
    serving, or whether Mac-specific quantized checkpoints need to be
    produced.

## Alternatives Considered

**Option A (chosen) — Independent deployment: different models on each machine.**
Pros: multiple models run simultaneously; huge models fit on the Mac; no memory
waste; independent failure domains; no distributed-system complexity; preserves
repo architecture. Cons: two runtime ecosystems to manage; no latency
optimization for single fit-able models; Mac Studio M5 Ultra specs unverified.

**Option B — EXO disaggregated prefill/decode.**
Pros: 2.8x latency speedup for a single fit-able model with large context
(blog-validated for dense 8B). Cons: both machines tied to one model (no
multi-model); weights duplicated on both machines (memory waste); capped at
smaller machine's memory (can't run huge models); MoE sparse activation reduces
prefill compute intensity (less overlap, less speedup); physically impossible
for frontier MoE (DGX Spark can't hold 380GB weight copy); either machine down
= model down; adds distributed-system complexity (EXO mesh, profiling, KV
streaming). Rejected — wrong trade-off for the operator's goals.

**Option C — Cross-host weight sharding (tensor/expert parallelism).**
Pros: could theoretically run models larger than either machine alone. Cons:
heterogeneous architecture (CUDA vs Metal) with no framework support; 10 GbE
too slow for per-layer all-reduce or all-to-all expert routing; no existing
tool (EXO, vLLM, SGLang, FreeToken) supports this across CUDA+Metal; violates
OOS "not a multi-host orchestrator." Rejected — not feasible.

**Option D — Mac Studio only, retire the DGX Spark.**
Pros: simplest (one machine, one runtime ecosystem). Cons: forfeits the DGX
Spark's 100 TFLOPs CUDA compute for dense models and GPTQ models; forfeits
vLLM/SGLang/FreeToken CUDA ecosystem; the DGX Spark is already owned and
operational. Rejected — the two machines are complementary, not redundant.

**Option E — EXO for latency-critical workloads + independent deployment for
multi-model.**
Pros: gets the best of both — EXO when you need one model fast, independent
when you need multiple models. Cons: doubles operational complexity (manage
both EXO mesh and independent scripts); EXO and independent deployment cannot
run simultaneously on the same machines (EXO ties up both); the operator would
need to stop independent models, start EXO, run the latency-critical request,
then restart independent models. Rejected — operational overhead exceeds the
benefit for a single-operator playground. Revisit if a latency-critical
workload emerges that justifies the mode-switching cost.

## To Investigate

- **Mac Studio M5 Ultra actual specs**: confirm 512GB unified memory, measure
  actual memory bandwidth (Apple's `memory_bandwidth` tool or benchmark), GPU
  TFLOPs. Compare to M3 Ultra baseline (819 GB/s, ~26 TFLOPs FP16).
- **MLX MoE support for GLM-5.2 753B**: does MLX support GLM-5.2's architecture?
  What quantization formats does MLX support for MoE? Is there an MLX-format
  NVFP4 or 4-bit checkpoint of GLM-5.2?
- **llama.cpp MoE performance on Apple Silicon**: benchmark GLM-5.2 753B GGUF
  on the Mac Studio with Metal backend. Compare tokens/sec to FreeToken hybrid
  on the DGX Spark.
- **FreeToken on Apple Silicon**: does FreeToken have an ARM/Metal backend, or
  is it CUDA-only? If ARM/Metal support exists or is planned, evaluate it as the
  Mac-side MoE runtime (semantic-aware caching for agent loops).
- **NVFP4 → MLX/GGUF conversion**: can NVFP4 checkpoints be converted to MLX
  or GGUF format? Or do Mac-specific quantized checkpoints need to be produced
  from the BF16 source?
- **Network topology**: is 10 GbE the actual link between the two machines?
  Could a direct Thunderbolt 4/5 link (40-80 Gbps) be used for lower-latency
  communication if EXO is ever revisited?
- **EXO MoE benchmarks**: has EXO published MoE benchmarks (not just dense
  Llama-3.1 8B)? If EXO demonstrates a speedup for MoE models that offsets the
  sparse-activation compute reduction, revisit Option E.
- **Apple's own serving framework**: is Apple developing a vLLM-equivalent for
  Apple Silicon (beyond MLX's inference API)? Any server-mode MLX with
  OpenAI-compatible API?
- **slotstream as Mac-side fallback**: slotstream
  ([github.com/carloslfu/slotstream](https://github.com/carloslfu/slotstream))
  is a Swift binary that disk-streams MoE experts from SSD on Apple Silicon.
  Currently single-model (Qwen3.8-Flash-Next 4-bit only), no tool calls, no
  images, no JSON-schema output, 32k context cap. Evaluate whether it adds
  support for GLM-5.2 or other MoE models by the time the Mac Studio ships.
  If it does, it becomes the fallback for models that exceed 512GB unified
  memory (where MLX's all-or-nothing loading fails). If it remains single-
  model, llama.cpp's mmap path is the fallback instead.
- **MLX partial-materialization workaround**: slotstream exists because MLX
  can't partially materialize MoE experts from mmap. Has Apple or the MLX
  team acknowledged this limitation? Is a fix planned? If MLX gains partial
  expert materialization, the need for slotstream or llama.cpp mmap on Macs
  that can't hold the full model is reduced.

## Validation

- [ ] Mac Studio M5 Ultra 512GB is released and specs match assumptions
      (≥512GB unified memory, ≥819 GB/s bandwidth)
- [ ] MLX or llama.cpp serves a small MoE model (Qwen3-Next-80B-A3B) on the Mac
      Studio with correct output
- [ ] GLM-5.2 753B (NVFP4 or GGUF 4-bit) loads and serves on the Mac Studio
      without OOM
- [ ] GLM-5.2 753B tokens/sec on the Mac Studio (unified memory) is
      significantly faster than FreeToken hybrid on the DGX Spark (disk
      streaming)
- [ ] DGX Spark and Mac Studio serve different models simultaneously without
      interference
- [ ] `test-query-local.sh -h ai-mac -p 8000` returns HTTP 200 from the Mac
      Studio
- [ ] `test-query-local.sh -h ai-dgx -p 8000` returns HTTP 200 from the DGX
      Spark (existing functionality, regression check)
- [ ] OOS doc amended to clarify two-independent-hosts is in-scope
- [ ] README model table updated with "Host" column and Mac Studio rows
- [ ] ADR-202608021744 cross-references this ADR for Mac Studio runtime selection
