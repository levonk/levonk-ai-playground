# NVIDIA HGX / NVLink (H100, H200, B200, B300)

> Systems with NVLink interconnect can shard models across GPUs efficiently.
> This is the platform to use for frontier models (400B+) that don't fit on a
> single GPU. This repo's scripts work on these systems but the default
> TP=1 assumption should be revisited.

## Specs by system

| System | GPUs | VRAM/GPU | Total VRAM | Interconnect | BW (bidirectional) |
|--------|------|----------|-----------|-------------|-------------------|
| HGX H100 | 8 × H100 | 80 GB HBM3 | 640 GB | NVLink 4 | 900 GB/s |
| HGX H200 | 8 × H200 | 141 GB HBM3e | 1,128 GB | NVLink 4 | 900 GB/s |
| HGX B200 | 8 × B200 | 192 GB HBM3e | 1,536 GB | NVLink 5 | 1,800 GB/s |
| HGX B300 | 8 × B300 | 288 GB HBM3e | 2,304 GB | NVLink 5 | 1,800 GB/s |

## When to use this platform

- **Frontier models** — Llama-3.1-405B, DeepSeek-R1-671B, and similar models
  that require >96 GB VRAM for weights alone
- **High-throughput sharded serving** — when you need a single large model to
  serve thousands of concurrent users with low latency
- **Training / fine-tuning** — NVLink enables efficient data-parallel and
  tensor-parallel training (out of scope for this repo)

## NVLink vs PCIe

| Metric | PCIe Gen 5 | NVLink 4 | NVLink 5 |
|--------|-----------|-----------|-----------|
| Bandwidth (bidirectional) | 128 GB/s | 900 GB/s | 1,800 GB/s |
| AllReduce latency | 12–25 μs | 1.5–3.0 μs | 1.0–2.0 μs |
| Relative speed | 1× | ~7× | ~14× |

NVLink makes TP>1 practical. AllReduce operations that take 12–25 μs on PCIe
take 1–3 μs on NVLink. For a model that does AllReduce on every layer of every
forward pass, this is the difference between usable and unusable sharding.

## Recommended strategies

### Large model, sharded (TP=N)

For models that don't fit on a single GPU:

```bash
TENSOR_PARALLEL_SIZE=8 ./scripts/model-chat.sh
```

On NVLink, TP=8 is efficient. The model's layers are split across all 8 GPUs,
and AllReduce operations complete in microseconds.

### Multiple medium models, isolated (TP=1)

For serving multiple 70B–27B models concurrently, use TP=1 per model and
pin each to a different GPU (same as the PCIe multi-GPU strategy):

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/model-chat.sh
CUDA_VISIBLE_DEVICES=1 ./scripts/model-code.sh
# ...
```

### Hybrid: one sharded large model + isolated small models

```bash
# GPUs 0-3: sharded 405B model
CUDA_VISIBLE_DEVICES=0,1,2,3 TENSOR_PARALLEL_SIZE=4 ./scripts/model-chat.sh

# GPU 4: code model (TP=1)
CUDA_VISIBLE_DEVICES=4 ./scripts/model-code.sh

# GPU 5: lite model (TP=1)
CUDA_VISIBLE_DEVICES=5 ./scripts/model-lite.sh
```

## Runtime selection

| Model type | Runtime | Why |
|-----------|---------|-----|
| Dense, sharded (TP>1) | vLLM | Most mature TP support over NVLink |
| Dense, TP=1 | SGLang | Better single-GPU throughput on Blackwell |
| MoE | FreeToken | MoE-optimized |
| GPTQ | vLLM | Only vLLM supports GPTQ |

## Cost vs benefit

NVLink systems are significantly more expensive than PCIe systems. If your
workload is high-concurrency serving of 70B-and-below models, a multi-GPU PCIe
system at TP=1 may deliver better price/performance. NVLink is justified when:

- You need to serve a single model larger than 96 GB (frontier models)
- You need the lowest possible latency for a sharded model
- You are training or fine-tuning (out of scope for this repo)

See [NVIDIA Multi-GPU PCIe](nvidia-multi-gpu-pcie.md) for the TP=1 alternative.
