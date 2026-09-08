# Lightweight Companion Models

> Why and how to run a small "lite" model alongside your primary model for
> routine tasks that don't need the full model's depth.
>
> Source: [lws.io — My local model setup](https://lws.io/blog/my-local-model-setup/)

## The problem

Running a 35B–80B model for every request is wasteful when the task is simple:
- "Rewrite this paragraph"
- "What does this error mean?"
- "Format this JSON"
- "Summarize this in one sentence"

These tasks don't need deep reasoning or large knowledge — a 4B model handles
them just as well, 10x faster, with negligible VRAM cost.

## The solution: big + lite

Run two models simultaneously:
- **Primary model** (80B or 35B) — for reasoning, coding, complex tasks
- **Lite model** (4B) — for formatting, simple Q&A, throwaway chats

The lite model uses ~2.5 GB of VRAM (AWQ-4bit), leaving the rest for the
primary model's KV cache. On GB10 128GB, the lite model is essentially free.

## Model selection

### Lite model criteria

- **Small** — <=4B parameters, <=3 GB VRAM in 4-bit
- **Fast** — high token generation speed (smaller = faster)
- **Instruction-tuned** — follows instructions well
- **Same family as primary** — consistent behavior and tokenizer

### This repo's lite model

| Platform | Model | VRAM | Port |
|----------|-------|------|------|
| NVIDIA GB10 | Qwen3-4B-Instruct-AWQ-4bit | ~2.5 GB | 8004 |
| Apple Silicon | mlx-community/Qwen3-4B-Instruct-4bit | ~2.5 GB | 8004 |
| Vulkan | Qwen3-4B-Instruct-GGUF | ~2.5 GB | 8004 |

Qwen3-4B is chosen because:
- Same family as the primary models (Qwen3-Next-80B-A3B)
- Very small (4B params, ~2.5 GB in 4-bit)
- Fast (small dense model, no MoE overhead)
- Good instruction following for its size

### Alternative lite models

| Model | Params | VRAM (4-bit) | Notes |
|-------|--------|-------------|-------|
| Qwen3-4B-Instruct | 4B | ~2.5 GB | This repo's default |
| Gemma-4-E4B-it | ~4B | ~2.4 GB | Used in lws.io blog |
| Llama-3.2-3B | 3B | ~1.8 GB | Very small, good quality |
| Qwen3-1.7B | 1.7B | ~1 GB | Ultra-light, for trivial tasks |

## When to use the lite model vs the primary

| Task | Use lite? | Why |
|------|-----------|-----|
| "Rewrite this paragraph" | Yes | No deep knowledge needed |
| "What does this error mean?" | Yes | Pattern matching, not reasoning |
| "Format this as JSON" | Yes | Formatting, not generation |
| "Summarize this in one sentence" | Yes | Compression, not analysis |
| "Write a function to sort a list" | No | Use code model |
| "Debug this algorithm" | No | Use reasoning model |
| "Explain quantum computing" | No | Use chat model (needs knowledge) |
| "Review this PR" | No | Use code or reasoning model |

The rule of thumb: if the task is about **transformation** (formatting,
rewriting, summarizing), use lite. If it's about **generation** (writing
code, reasoning, explaining), use the primary model.

## Client configuration

Point your client at the appropriate port:

```bash
# Lite model for simple tasks
curl http://localhost:8004/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Format this as JSON: ..."}]}'

# Primary model for complex tasks
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-next-80b","messages":[{"role":"user","content":"Debug this function: ..."}]}'
```

Most OpenAI-compatible clients (Apollo, Raycast, Hermes, etc.) let you
configure multiple endpoints. Set up the lite model as the default and
switch to the primary when needed.

## Performance comparison

On GB10 128GB:

| Model | Weights | Token gen speed | VRAM for KV cache |
|-------|---------|----------------|-------------------|
| Qwen3-Next-80B-A3B (primary) | ~45 GB | ~30-40 tok/s | ~77 GB |
| Qwen3-4B (lite) | ~2.5 GB | ~150+ tok/s | ~119 GB |

The lite model generates tokens 4-5x faster than the 80B model. For simple
tasks, this means near-instant responses.

## On constrained hardware (Apple Silicon 16GB)

On a 16 GB Mac, you can't run an 80B model at all. But you can run:
- Qwen3-4B 4-bit (~2.5 GB) + OS overhead (~6 GB) = 8.5 GB used
- Leaves ~7.5 GB for KV cache

This makes the lite model a viable primary model on constrained hardware.
On 16 GB Macs, `model-lite.sh` can serve as the only model.
