# Platform Reference Index

> Hardware-specific deployment guides. Each platform doc covers VRAM budget,
> recommended runtimes, tensor-parallelism strategy, and co-location rules for
> that hardware class. Read the one that matches your machine before choosing
> models or running scripts.

## Platforms

| Platform | VRAM | Interconnect | Primary runtime | Details |
|----------|------|--------------|-----------------|---------|
| NVIDIA GB10 128GB (DGX Spark) | 128 GB | single GPU | SGLang / FreeToken | [details](nvidia-gb10-128gb.md) |
| Apple Silicon Mac (M1–M5) | 16–192 GB unified | unified memory | MLX (vllm-mlx) | [details](apple-silicon.md) |
| NVIDIA multi-GPU PCIe (e.g. 8x RTX PRO 6000) | 96 GB × N | PCIe Gen 5 | vLLM / SGLang (TP=1) | [details](nvidia-multi-gpu-pcie.md) |
| NVIDIA HGX / NVLink (H100, H200, B200, B300) | 80–192 GB × N | NVLink 4/5 | vLLM / SGLang (TP=N) | [details](nvidia-hgx-nvlink.md) |

## How to use these docs

1. **Identify your platform.** Run `detect_system` from `runner-lib.sh` or check
   the table above. If your hardware spans categories (e.g. a Mac Studio with
   an external eGPU), read both docs and reconcile.

2. **Check the VRAM budget.** Each platform doc lists how much VRAM is available
   for model weights vs. KV cache after OS/runtime overhead. Use this to decide
   which models fit and how many can co-locate.

3. **Check the tensor-parallelism recommendation.** Multi-GPU PCIe systems should
   prefer TP=1 (isolated serving) unless sharding is unavoidable. NVLink systems
   can shard freely. Single-GPU systems are always TP=1.

4. **Check co-location rules.** Each platform doc lists which model combinations
   can run simultaneously within the VRAM budget. See also
   [Model Co-location Budget](../guides/model-colocation-budget.md).

5. **Cross-reference the guides.** Platform docs link to general guides for
   topics that apply across all platforms (KV cache planning, quantization,
   SSD wear, networking, etc.).

## General guides

Guides that apply to all platforms live in [`../guides/`](../guides/INDEX.md).
Platform docs reference them where relevant.

## Sources

- [lws.io — My local model setup](https://lws.io/blog/my-local-model-setup/) — Apple Silicon Mac mini setup, MoE model selection, Tailscale networking
- [GPUpartner — A practical guide to running 8x RTX PRO 6000's](https://www.gpupartner.com/blog/a-practical-guide-to-running-8x-rtx-pro-6000s) — multi-GPU PCIe capacity planning, TP=1 vs sharding, KV cache density
