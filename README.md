# vLLM Model Runner Scripts

Primarily tracks top performers from https://github.com/AlexsJones/llmfit for Nvidia GB10 128GB, CUDA w/ vLLM

> **AI agents**: see [`AGENTS.md`](AGENTS.md) for binding project context, conventions, and boundaries before editing this repo.

## Architecture

This project uses a **shared library pattern** for DRY (Don't Repeat Yourself) code organization:

- **`scripts/`** — user-facing entrypoints: one `model-*.sh` runner per model, plus the `test-query*.sh` smoke-test tools
- **`libexec/`** — internals, never called directly: `runner-lib.sh` (shared library every runner sources) and `system-one-server.py` (the scorer service)
- **`tests/`** — smoke tests for endpoints the generic test-query tools can't exercise (e.g. `test-system-one.sh` for the scorer)

### Why Shell Library Instead of Ansible?

For this use case (local development playground, single-machine Docker deployment), a shell library is the appropriate choice:

**Shell Library Advantages:**
- Simple and lightweight - no additional dependencies
- Direct control over Docker commands
- Easy to debug and modify
- Fast execution with minimal overhead
- Perfect for local development workflows

**Ansible Would Be Overkill Because:**
- Designed for multi-server infrastructure orchestration
- Adds complexity for single-machine local development
- Requires Python and Ansible installation
- Overhead of inventory management and playbooks
- Better suited for production deployments across multiple hosts

## Models

Each model runs on a unique host port to allow simultaneous deployment:

| Script (in `scripts/`) | Model | Account | Host Port | VRAM | Special Config |
|--------|-------|---------|-----------|------|----------------|
| `model-chat.sh` | Qwen3-Next-80B-A3B-Instruct-AWQ-4bit | cyankiwi | 8000 | ~45 GB | `--quantization awq_merlin` |
| `model-code.sh` | Qwen3-Coder-Next-AWQ-4bit | cyankiwi | 8001 | ~45 GB | `--quantization awq_merlin` |
| `model-reason.sh` | Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-int4 | codgician | 8002 | ~20 GB | GPTQ (vLLM fallback) |
| `model-general.sh` | Qwen3-Next-80B-A3B-Thinking-AWQ-4bit | cyankiwi | 8003 | ~45 GB | `--max-model-len 262144`, speculative decoding |
| `model-lite.sh` | Qwen3-4B-Instruct-AWQ-4bit | Qwen | 8004 | ~2.5 GB | Lightweight companion for simple tasks |
| `model-tts.sh` | Kokoro-82M | hwdsl2 | 8005 | ~0.3 GB | OpenAI-compatible `/v1/audio/speech` |
| `model-stt.sh` | Whisper large-v3-turbo | hwdsl2 | 8006 | ~1.5 GB | OpenAI-compatible `/v1/audio/transcriptions` |
| `model-system-one.sh` | system-one-qwen3.5-4b-scorer-v2b (LoRA on Qwen3.5-4B-Base) | pngwn | 8007 | ~9 GB bf16 | Native Python scorer (torch+PEFT), `POST /score` — not a chat endpoint |

VRAM figures are approximate for AWQ-4bit / GPTQ-int4 quantization on the
NVIDIA GB10 128GB. See [`internal-docs/platforms/`](internal-docs/platforms/INDEX.md)
for platform-specific sizing and [`internal-docs/guides/model-colocation-budget.md`](internal-docs/guides/model-colocation-budget.md)
for which models can run simultaneously.

## Usage

### Basic Usage

Run a specific model:

```bash
./scripts/model-chat.sh        # 80B chat model (port 8000)
./scripts/model-code.sh        # 80B code model (port 8001)
./scripts/model-reason.sh      # 35B reasoning model (port 8002)
./scripts/model-general.sh     # 80B thinking model (port 8003)
./scripts/model-lite.sh        # 4B lightweight model (port 8004)
./scripts/model-tts.sh         # Kokoro TTS (port 8005)
./scripts/model-stt.sh         # Whisper STT (port 8006)
./scripts/model-system-one.sh  # System One decision scorer (port 8007)
```

### Custom Configuration

Override configuration via environment variables:

```bash
# Change port
HOST_PORT=9000 ./scripts/model-chat.sh

# Change model
MODEL=custom-model-name ./scripts/model-chat.sh

# Change account
ACCT=my-account ./scripts/model-chat.sh

# GPU memory utilization
GPU_MEMORY_UTILIZATION=0.95 ./scripts/model-chat.sh

# Tensor parallel size (for multi-GPU)
TENSOR_PARALLEL_SIZE=2 ./scripts/model-chat.sh
```

### Running Multiple Models Simultaneously

Since each model uses a unique port, you can run them simultaneously:

```bash
# Terminal 1 — chat model (~45 GB VRAM)
./scripts/model-chat.sh

# Terminal 2 — code model (~45 GB VRAM)
./scripts/model-code.sh

# Terminal 3 — reasoning model (~20 GB VRAM)
./scripts/model-reason.sh

# Terminal 4 — lite model (~2.5 GB VRAM, runs alongside anything)
./scripts/model-lite.sh

# Terminal 5 — TTS (~0.3 GB VRAM)
./scripts/model-tts.sh

# Terminal 6 — STT (~1.5 GB VRAM)
./scripts/model-stt.sh

# Terminal 7 — System One scorer (~9 GB bf16, single-pass option scoring)
./scripts/model-system-one.sh
```

See [`internal-docs/guides/model-colocation-budget.md`](internal-docs/guides/model-colocation-budget.md)
for which combinations fit within your VRAM budget.

## Test Scripts

Two helper scripts in `scripts/` send a quick prompt to a running model server
and print the response, useful for smoke-testing a deployment after
`model-*.sh` brings it up. Endpoint-specific tests that can't use the chat
API live in `tests/` — currently `tests/test-system-one.sh` for the scorer
(`GET /health` + `POST /score` on port 8007).

### `test-query.sh` — local Docker discovery

Auto-discovers running Docker containers whose image name contains `llm` and
POSTs an OpenAI-style `/v1/chat/completions` request to each, reporting HTTP
status and the returned content (or reasoning, or raw body). When no
containers are found and no `-p` is given, it falls back to host `ai` port
`8002` (overridable via `FALLBACK_HOST` / `FALLBACK_PORT`).

```bash
./scripts/test-query.sh                       # ping every running LLM container
./scripts/test-query.sh 'hello world'         # custom prompt, all containers
./scripts/test-query.sh -p 8002 -m qwen3 'hi' # single port, explicit model
```

| Option            | Default                                                              |
|-------------------|----------------------------------------------------------------------|
| `PROMPT`          | `ping`                                                               |
| `-p, --port PORT` | auto-discover from Docker                                            |
| `--HOST HOST`     | `ai` (host for manual/fallback target; overrides `FALLBACK_HOST`)    |
| `-m, --model`     | auto-detect from `/v1/models`, fall back to container name           |
| `FALLBACK_HOST`   | `ai` (used only when discovery finds nothing, no `-p`, no `--HOST`)  |
| `FALLBACK_PORT`   | `8002` (used only when discovery finds nothing and no `-p` given)    |
| `CONNECT_TIMEOUT` | `5` (curl `--connect-timeout`, seconds)                              |
| `MAX_TIME`        | `120` (curl `--max-time`, seconds)                                   |
| `MAX_TOKENS`      | `2048` (`max_tokens` in the request payload)                         |

### `test-query-local.sh` — remote host

Targets a single OpenAI-compatible endpoint on a remote machine (no Docker
discovery). Defaults to host `ai` port `8002`, so it runs with no arguments.

```bash
./scripts/test-query-local.sh                         # ping ai:8002
./scripts/test-query-local.sh 'hello world'           # custom prompt, ai:8002
./scripts/test-query-local.sh -p 8002 'hello world'   # explicit port
./scripts/test-query-local.sh -h 10.0.0.5 -p 8001 -m qwen3 hi
```

| Option            | Default                                                              |
|-------------------|----------------------------------------------------------------------|
| `PROMPT`          | `ping`                                                               |
| `-p, --port PORT` | `8002`                                                               |
| `-h, --host HOST` | `$REMOTE_HOST` or `ai`                                               |
| `--HOST HOST`     | same as `--host` (overrides the `ai` default)                        |
| `-m, --model`     | auto-detect from `/v1/models`, fall back to `default`                |
| `CONNECT_TIMEOUT` | `5` (curl `--connect-timeout`, seconds)                              |
| `MAX_TIME`        | `120` (curl `--max-time`, seconds)                                   |
| `MAX_TOKENS`      | `2048` (`max_tokens` in the request payload)                         |

Both scripts use `jq` when available and fall back to `sed`-based escaping
(`test-query.sh` only). On failure they print the curl exit code mapped to a
human message (connection refused, DNS failure, timeout, empty reply, etc.),
the underlying `curl` stderr, and — for non-200 HTTP responses — any server
error message from the JSON body plus the first 500 bytes. See the script
headers for full usage.

## Environment Variables

All scripts support these environment variables:

- `ACCT` - Hugging Face account name
- `MODEL` - Model name
- `HOST_PORT` - Host port to expose (container always uses 8000 internally)
- `LOCAL_CACHE_DIR` - Local cache directory (default: `/root/.cache`)
- `CONTAINER_CACHE_DIR` - Container cache directory (default: `/root/.cache`)
- `TENSOR_PARALLEL_SIZE` - Number of GPUs for tensor parallelism (default: 1)
- `GPU_MEMORY_UTILIZATION` - GPU memory utilization ratio (default: 0.90)
- `MAX_NUM_BATCHED_TOKENS` - Maximum batched tokens (default: 32768)
- `KV_CACHE_DTYPE` - KV cache quantization dtype (default: empty/BF16). Set to `fp8` to enable FP8 KV cache, which halves KV cache memory and doubles concurrent session capacity with minimal quality impact. See [`internal-docs/guides/fp8-kv-cache.md`](internal-docs/guides/fp8-kv-cache.md).
- `HUGGING_FACE_HUB_TOKEN` - HF token for private models (loaded from `.env`)
- `OTLP_TRACES_ENDPOINT` - OTLP endpoint for OpenTelemetry tracing (e.g. `http://openlit:4318`). When set, vLLM gets `--otlp-traces-endpoint` and accepts incoming `traceparent` headers for distributed tracing with LiteLLM. Empty/unset = tracing disabled.

### TTS/STT Environment Variables

- `TTS_IMAGE` - Override the Kokoro TTS Docker image (default: `hwdsl2/kokoro-server:cuda` on NVIDIA, `hwdsl2/kokoro-server` on CPU)
- `STT_IMAGE` - Override the Whisper STT Docker image (default: `hwdsl2/whisper-server:cuda` on NVIDIA, `hwdsl2/whisper-server` on CPU)
- `TTS_CONTAINER_NAME` - Container name for TTS (default: `kokoro-tts`)
- `STT_CONTAINER_NAME` - Container name for STT (default: `whisper-stt`)
- `CUSTOM_CONTAINER_PORT` - Internal container port (auto-set by TTS/STT scripts)
- `CUSTOM_IMAGE` - Override any custom container image (generic override)
- `KOKORO_LOCAL_ONLY` - Set to `1` for Kokoro air-gapped mode (pre-cached model only)
- `WHISPER_MODEL` - Whisper model name (default: `large-v3-turbo`; alternatives: `base`, `small`, `medium`, `large-v3`)
- `WHISPER_DEVICE` - Compute device: `cpu`, `cuda`, or `auto` (default: `auto`)
- `WHISPER_LOCAL_ONLY` - Set to `1` for Whisper air-gapped mode (pre-cached model only)
- `WHISPER_LANGUAGE` - Language code (default: `auto`; e.g. `en`, `ja`, `zh`)

### System One Scorer

`model-system-one.sh` deploys `pngwn/system-one-qwen3.5-4b-scorer-v2b`, a
"System One"-style decision scorer. Two things about it are easy to get
wrong:

- **It is not a smaller model.** The artifact on HF is a LoRA adapter
  (~122 MB, ~30M trained weights) that must be loaded on top of the full
  `Qwen/Qwen3.5-4B-Base` checkpoint. Serving it still requires the entire
  4B backbone (~9 GB in bf16).
- **Its speed comes from the access pattern, not the size.** Each option of
  a question gets a scalar score in a *single forward pass* (~128 ms per
  5-option query) — there is no autoregressive generation. The output is a
  calibrated probability distribution over exactly the options you supply,
  which makes it useful for routing/triage decisions where an LLM call
  would be slower and uncalibrated.

This deployment has **two files**, unlike the LLM scripts:

- `scripts/model-system-one.sh` — the runner script (repo convention: one
  `model-*.sh` per model). Detects the system, checks Python deps, and
  launches the server on port 8007.
- `libexec/system-one-server.py` — the actual HTTP service (`POST /score`), a small
  stdlib server it launches via `run_native`. It exists because the scorer's
  classification head lives in the adapter's `modules_to_save`, which
  SGLang/vLLM/FreeToken/MLX do not load — those runtimes would serve the
  base model as a text generator, not a scorer. A native
  torch+transformers+peft process is the only way to serve the real
  scoring semantics without building a custom image.

`POST /score` accepts `{"state", "question", "options": [...]}` and returns
per-option raw scores plus temperature-calibrated probabilities. Requires
`pip install torch transformers peft` on the host. The adapter is
CC-BY-NC-4.0 (non-commercial use only).

- `S1_BASE_MODEL` - Base checkpoint (default: `Qwen/Qwen3.5-4B-Base`)
- `S1_ADAPTER` - PEFT adapter repo (default: `pngwn/system-one-qwen3.5-4b-scorer-v2b`)
- `S1_MAX_LEN` - Max sequence length (default: `384`, matches training; state is truncated first, never the question/option)
- `S1_TEMPERATURE` - Calibration temperature (default: `2.35`, fitted on val; request `"temperature": 0` for raw scores)
- `S1_BATCH_SIZE` - Options per forward pass (default: `8`)
- `S1_MAX_OPTIONS` - Max options per request (default: `64`)
- `S1_DEVICE` - `cuda`, `mps`, `cpu`, or `auto` (default: `auto`)
- `S1_DTYPE` - `bf16`, `fp16`, `fp32` (default: bf16 on GPU/MPS, fp32 on CPU)
- `S1_CONTAINER_NAME` - Process match name for stop logic (default: `system-one-server.py`)

## Adding a New Model

Create a new script in `scripts/` following this pattern (see `scripts/model-chat.sh`
for a full example, or `scripts/model-lite.sh` for a simpler one):

```bash
#!/usr/bin/env bash
#
# Your Model Runner
# Auto-detects system and selects the appropriate model + runtime.
#
# NVIDIA (DGX Spark GB10):  Your-Model-AWQ-4bit  (SGLang)
# Apple Silicon (Mac):      mlx-community Your-Model-4bit  (MLX)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../libexec/runner-lib.sh
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/libexec/runner-lib.sh"

# Per-system model configuration
configure_model() {
  local system
  system="$(detect_system)"
  echo "Detected system: $system"

  case "$system" in
    nvidia)
      ACCT="${ACCT:-your-account}"
      MODEL="${MODEL:-your-model-name}"
      HOST_PORT="${HOST_PORT:-8007}"  # Use a unique port
      RUNTIME="${RUNTIME:-sglang}"
      ;;
    apple-silicon)
      ACCT="${ACCT:-mlx-community}"
      MODEL="${MODEL:-your-model-name-4bit}"
      HOST_PORT="${HOST_PORT:-8007}"
      RUNTIME="${RUNTIME:-mlx}"
      ;;
    *)
      echo "Error: Unsupported system '$system'." >&2
      exit 1
      ;;
  esac
}

main() {
  load_env
  configure_model
  check_prerequisites
  setup_huggingface
  clear_caches
  stop_existing_container "$MODEL"

  run_container "$ACCT" "$MODEL" "$HOST_PORT"
}

main "$@"
```

For TTS/STT or other non-LLM services, use `run_custom_container()` instead
of `run_container()` — see `scripts/model-tts.sh` and `scripts/model-stt.sh` for
examples. For services that can't run in any container runtime at all (like
the System One scorer, a PEFT classifier), ship a small server in `libexec/` and
launch it via `run_native()` — see `scripts/model-system-one.sh`.

## Shared Library Functions

`libexec/runner-lib.sh` provides:

- `load_env()` - Load `.env` file if present
- `check_prerequisites()` - Verify Docker/sudo (Docker runtimes) or CLI (native runtimes)
- `setup_huggingface()` - Authenticate with Hugging Face
- `clear_caches()` - Clear system caches (requires sudo on Linux, `purge` on macOS)
- `stop_existing_container()` - Stop running container or process if exists
- `detect_system()` - Auto-detect hardware type (nvidia, apple-silicon, vulkan, cpu)
- `detect_runtime()` - Auto-select runtime based on detected system
- `run_container()` - Dispatch to the selected runtime (sglang, vllm, freetoken, mlx)
- `run_sglang_container()` - Run SGLang in Docker (dense models on GB10)
- `run_vllm_container()` - Run vLLM in Docker (GPTQ, compressed-tensors, multi-GPU PP)
- `run_freetoken()` - Run FreeToken as native Python (MoE models)
- `run_mlx()` - Run vllm-mlx as native Python (Apple Silicon)
- `run_custom_container()` - Run a custom Docker image (TTS, STT, other services)
- `run_native()` - Run a native Python process (TTS/STT on Apple Silicon/CPU)

## Documentation

### Platform Reference

Platform-specific deployment guides with VRAM budgets, runtime
recommendations, and co-location rules:

- [`internal-docs/platforms/INDEX.md`](internal-docs/platforms/INDEX.md) — overview
- [`internal-docs/platforms/nvidia-gb10-128gb.md`](internal-docs/platforms/nvidia-gb10-128gb.md) — GB10 (primary target)
- [`internal-docs/platforms/apple-silicon.md`](internal-docs/platforms/apple-silicon.md) — Apple Silicon Macs
- [`internal-docs/platforms/nvidia-multi-gpu-pcie.md`](internal-docs/platforms/nvidia-multi-gpu-pcie.md) — Multi-GPU PCIe (TP=1 strategy)
- [`internal-docs/platforms/nvidia-hgx-nvlink.md`](internal-docs/platforms/nvidia-hgx-nvlink.md) — HGX/NVLink (frontier models)

### General Guides

Topic guides that apply across all platforms:

- [`internal-docs/guides/INDEX.md`](internal-docs/guides/INDEX.md) — overview
- [`internal-docs/guides/kv-cache-capacity-planning.md`](internal-docs/guides/kv-cache-capacity-planning.md) — KV cache sizing formulas and tables
- [`internal-docs/guides/fp8-kv-cache.md`](internal-docs/guides/fp8-kv-cache.md) — FP8 KV cache (doubles concurrent capacity)
- [`internal-docs/guides/model-colocation-budget.md`](internal-docs/guides/model-colocation-budget.md) — Which models can run simultaneously
- [`internal-docs/guides/tensor-parallelism-guide.md`](internal-docs/guides/tensor-parallelism-guide.md) — TP=1 vs TP>1 decision framework
- [`internal-docs/guides/model-identifier-guide.md`](internal-docs/guides/model-identifier-guide.md) — How to read model names
- [`internal-docs/guides/quantization-guide.md`](internal-docs/guides/quantization-guide.md) — AWQ, GPTQ, OptiQ, FP8, BF16 trade-offs
- [`internal-docs/guides/lightweight-companion-models.md`](internal-docs/guides/lightweight-companion-models.md) — Why and how to use a lite model
- [`internal-docs/guides/ssd-wear-mitigation.md`](internal-docs/guides/ssd-wear-mitigation.md) — Protecting Mac SSDs with external storage
- [`internal-docs/guides/networking-guide.md`](internal-docs/guides/networking-guide.md) — Tailscale/tailnet for multi-device access

### FP8 KV Cache

Enable FP8 KV cache to double concurrent session capacity with minimal
quality impact (~1-2 benchmark points vs BF16):

```bash
# Enable for any LLM model script:
KV_CACHE_DTYPE=fp8 ./scripts/model-chat.sh

# Or add to .env for all scripts:
echo 'KV_CACHE_DTYPE=fp8' >> .env
```

See [`internal-docs/guides/fp8-kv-cache.md`](internal-docs/guides/fp8-kv-cache.md) for details.

### SSD Wear Mitigation (Apple Silicon)

Apple Silicon Macs have soldered, non-replaceable SSDs. Move model storage
to an external NVMe SSD to reduce write wear:

```bash
# In .env:
LOCAL_CACHE_DIR=/Volumes/ExternalSSD/huggingface
```

See [`internal-docs/guides/ssd-wear-mitigation.md`](internal-docs/guides/ssd-wear-mitigation.md) for the full strategy.
