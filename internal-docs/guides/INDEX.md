# Guides Index

> General guides that apply across all platforms. Each guide covers a single
> topic. Platform docs in [`../platforms/`](../platforms/INDEX.md) reference
> these where relevant.

## Guides

| Guide | Topic | Source |
|-------|-------|--------|
| [KV Cache Capacity Planning](kv-cache-capacity-planning.md) | Formulas and tables for KV cache sizing | GPUpartner |
| [FP8 KV Cache](fp8-kv-cache.md) | FP8 KV cache quantization — doubles concurrent capacity | GPUpartner |
| [Model Co-location Budget](model-colocation-budget.md) | Which models can run simultaneously within a VRAM budget | GPUpartner + lws.io |
| [Tensor Parallelism Guide](tensor-parallelism-guide.md) | TP=1 vs TP>1 decision framework | GPUpartner |
| [Model Identifier Guide](model-identifier-guide.md) | How to read model names (params, MoE, quantization) | lws.io |
| [Quantization Guide](quantization-guide.md) | AWQ, GPTQ, OptiQ, FP8, BF16 — trade-offs and compatibility | lws.io + GPUpartner |
| [Lightweight Companion Models](lightweight-companion-models.md) | Why and how to run a small model alongside a big one | lws.io |
| [SSD Wear Mitigation](ssd-wear-mitigation.md) | Protecting soldered Mac SSDs — external storage strategy | Research |
| [Networking Guide](networking-guide.md) | Tailscale/tailnet for multi-device model access | lws.io |
