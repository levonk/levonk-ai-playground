#!/usr/bin/env bash
#
# LLM Model Runner Library
# Shared functions for deploying LLM models across multiple runtimes
#
# Supports five inference runtimes (per ADR-202608021744 as amended 2026-08-26):
#   - SGLang (cu130)   — default for in-VRAM dense models on GB10/Blackwell (Docker)
#   - FreeToken        — default for MoE models, in-VRAM and frontier (native Python)
#   - vLLM             — fallback for GPTQ/compressed-tensors/multi-GPU PP (Docker)
#   - MLX (vllm-mlx)   — Apple Silicon Mac serving via Metal/MLX (native Python)
#   - Strata           — llama.cpp-lineage engine tuned for DGX Spark (GB10,
#                        sm_121, aarch64); serves GGUF packs with MoE expert
#                        mmap (native CLI, no Docker)
#
# SGLang 1.0 landscape (announced 2026-11-12, tracker sgl-project/sglang#50045):
#   - SGLang Diffusion is bundled in the same package/CLI (image, video, 3D
#     generation + OpenAI-compatible image/video API + ComfyUI plugin).
#   - Paged Experts (v0.5.23, opt-in) runs MoE larger than VRAM on one GPU —
#     contests FreeToken's frontier-MoE slot; not yet validated on GB10.
#   - 1.0 removes sgl-model-gateway (use sgl-router), removes deprecated
#     flags (--disable-cuda-graph, --enable-dp-attention — unused here),
#     requires --admin-api-key for admin routes, and errors on unsupported
#     configs instead of silently adjusting them.
#
# See ADR-202608021744 for the tiered runtime selection rationale.
# See adr-20260826-freetoken-runtime-amendment.md for the FreeToken amendment.
# See internal-docs/guides/runtime-selection.md for the scenario→runner matrix.
#
# NOTE on Basalt (github.com/jesdga95/basalt): NOT supported here. Basalt is
# a Strata fork optimized for NVIDIA Blackwell but builds Linux/Windows
# x86-64 only (AVX2) and targets sm_120 — the DGX Spark GB10 is aarch64 +
# sm_121, so Basalt cannot run on it. Strata (its upstream) ships a
# linux-aarch64 engine with GB10-tuned defaults and is the correct choice.
#

set -euo pipefail

# =============================================================================
# Runtime Selection — change this to swap the inference engine
# =============================================================================
# Set RUNTIME to one of: "sglang", "freetoken", "vllm", "mlx", "strata".
# All scripts that source this library and call run_container() will use the
# selected runtime.
#
# Individual scripts can override by exporting RUNTIME before sourcing this
# file, e.g.:
#
#     RUNTIME=vllm source "$REPO_ROOT/libexec/runner-lib.sh"
#
# Per ADR-202608021744 (amended):
#   - SGLang is the default for in-VRAM dense models on GB10
#   - FreeToken is the default for MoE models (both in-VRAM and frontier)
#   - vLLM is the fallback for GPTQ, compressed-tensors, multi-GPU PP, and
#     models SGLang/FreeToken do not yet support
#   - MLX is for Apple Silicon Mac hosts (Mac Studio, MacBook Pro, etc.)
#
# NOTE: Model-specific args passed to run_container() (e.g. --max-model-len,
# --speculative-config) are runtime-specific.  When switching runtimes you
# may need to adjust those args in the individual model-*.sh scripts.
#
# NOTE: SGLang and vLLM run in Docker containers (GPU host).  FreeToken and
# MLX run as native Python processes (no Docker).  The dispatcher handles
# this transparently — callers always use run_container() regardless of
# runtime.
# =============================================================================
# RUNTIME is resolved after detect_system()/detect_runtime() are defined below.
# Scripts can override by setting RUNTIME before sourcing this file.

# Pinned image tags (do NOT use floating "latest" tags in production)
SGLANG_IMAGE="lmsysorg/sglang:v0.5.23-cu130"
VLLM_IMAGE="nvcr.io/nvidia/vllm:26.04-py3"

# FreeToken and MLX are native Python — no Docker image.
# Pin to a specific PyPI version or git tag in the model script or .env.
FREETOKEN_VERSION="${FREETOKEN_VERSION:-}"
MLX_VERSION="${MLX_VERSION:-}"

# Strata is a native CLI (signed engine feed) — no Docker image.
# Install: https://github.com/Niko1221/Strata (one-line installer, no sudo).
STRATA_VERSION="${STRATA_VERSION:-}"

# Default configuration
DEFAULT_LOCAL_CACHE_DIR="/root/.cache"
DEFAULT_CONTAIN_CACHE_DIR="/root/.cache"
DEFAULT_TENSOR_PARALLEL_SIZE=1
DEFAULT_GPU_MEMORY_UTILIZATION=0.90
DEFAULT_MAX_NUM_BATCHED_TOKENS=32768

# KV cache dtype — set to "fp8" to enable FP8 KV cache quantization.
# This halves KV cache memory usage, doubling concurrent session capacity
# with minimal quality impact (~1-2 benchmark points vs BF16).
# See internal-docs/guides/fp8-kv-cache.md for details.
# Supported values: "" (auto/BF16), "fp8", "fp8_e4m3", "fp8_e5m2"
# Note: SGLang support varies by version; vLLM supports fp8 directly.
DEFAULT_KV_CACHE_DTYPE=""

# =============================================================================
# System Detection — auto-detect GPU/hardware type and pick a default runtime
# =============================================================================
# detect_system() prints one of:
#   nvidia          — NVIDIA GPU present (DGX Spark, workstations with CUDA)
#   apple-silicon   — Apple Silicon Mac (M1/M2/M3/M4/M5, unified memory)
#   vulkan          — Vulkan-capable GPU (AMD, Intel) but no NVIDIA
#   cpu             — no GPU detected, CPU-only fallback
#
# detect_runtime() prints the default runtime for the detected system:
#   nvidia        → sglang   (dense default per ADR)
#   apple-silicon → mlx      (MLX via vllm-mlx)
#   vulkan        → vllm     (vLLM with Vulkan backend, or llama.cpp)
#   cpu           → vllm     (CPU fallback — slow but works)
#
# Scripts can override either by exporting RUNTIME or SYSTEM before sourcing.
detect_system() {
  # Allow explicit override
  if [ -n "${SYSTEM:-}" ]; then
    echo "$SYSTEM"
    return
  fi

  # NVIDIA: check for nvidia-smi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "nvidia"
    return
  fi

  # Apple Silicon: check for macOS + Apple chip
  if [ "$(uname)" = "Darwin" ]; then
    local cpu_brand
    cpu_brand=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
    if echo "$cpu_brand" | grep -qi "Apple"; then
      echo "apple-silicon"
      return
    fi
  fi

  # Vulkan: check for vulkaninfo
  if command -v vulkaninfo >/dev/null 2>&1; then
    echo "vulkan"
    return
  fi

  # Fallback: CPU-only
  echo "cpu"
}

detect_runtime() {
  local system="${1:-$(detect_system)}"
  case "$system" in
    nvidia)        echo "sglang" ;;
    apple-silicon) echo "mlx" ;;
    vulkan)        echo "vllm" ;;
    cpu)           echo "vllm" ;;
    *)             echo "vllm" ;;
  esac
}

# Resolve RUNTIME lazily — do NOT bake in the auto-detected default at source
# time. Model scripts set their own per-system default in configure_model()
# (which runs AFTER this file is sourced) via RUNTIME="${RUNTIME:-<default>}".
# If RUNTIME were resolved here, the auto-detected value would always win over
# the script's default, and e.g. model-general.sh's intended vLLM would
# silently run SGLang instead. resolve_runtime() fills in the auto-detected
# value only when neither the user nor the script set one; the dispatch
# functions call it before consulting $RUNTIME.
resolve_runtime() {
  if [ -z "${RUNTIME:-}" ]; then
    RUNTIME="$(detect_runtime)"
  fi
}

# =============================================================================
# Per-system model configuration helper
# =============================================================================
# configure_for_system() sets ACCT, MODEL, HOST_PORT, and RUNTIME based on
# the detected system. Each model script defines a configure_model() function
# with a case statement per system, then calls this helper.
#
# Pattern (in model-*.sh):
#
#   configure_model() {
#     case "$(detect_system)" in
#       nvidia)
#         ACCT="cyankiwi"
#         MODEL="Qwen3-Next-80B-A3B-Instruct-AWQ-4bit"
#         HOST_PORT=8000
#         RUNTIME="sglang"
#         ;;
#       apple-silicon)
#         ACCT="mlx-community"
#         MODEL="Qwen3-Next-80B-A3B-Instruct-4bit"
#         HOST_PORT=8000
#         RUNTIME="mlx"
#         ;;
#       *)
#         echo "Unsupported system for this model" >&2
#         exit 1
#         ;;
#     esac
#   }
#
# The script's main() calls configure_model() before run_container().
# =============================================================================

# Load environment variables from .env if it exists. Looks in the repo root
# (this file's parent directory) first so scripts/ entrypoints work from any
# working directory, then falls back to the caller's cwd.
load_env() {
  local lib_dir env_file
  lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  env_file="$lib_dir/../.env"
  if [ -f "$env_file" ]; then
    # shellcheck source=/dev/null
    source "$env_file"
  elif [ -f .env ]; then
    # shellcheck source=/dev/null
    source .env
  fi
}

# Check prerequisites
check_prerequisites() {
  resolve_runtime
  local missing=()

  # Docker-based runtimes (sglang, vllm) need docker + sudo
  if [ "$RUNTIME" = "sglang" ] || [ "$RUNTIME" = "vllm" ]; then
    command -v docker >/dev/null 2>&1 || missing+=("docker")
    command -v sudo >/dev/null 2>&1 || missing+=("sudo")

    if [ ${#missing[@]} -gt 0 ]; then
      echo "Error: Missing required commands for RUNTIME=$RUNTIME: ${missing[*]}"
      exit 1
    fi

    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
      echo "Error: Docker is not running. Please start Docker and try again."
      exit 1
    fi
  fi

  # FreeToken needs the `ft` CLI
  if [ "$RUNTIME" = "freetoken" ]; then
    if ! command -v ft >/dev/null 2>&1; then
      echo "Error: FreeToken 'ft' CLI not found."
      echo "Install with: uv pip install \"freetoken[accel]\""
      exit 1
    fi
  fi

  # MLX needs the `vllm-mlx` CLI (Apple Silicon only)
  if [ "$RUNTIME" = "mlx" ]; then
    if ! command -v vllm-mlx >/dev/null 2>&1; then
      echo "Error: vllm-mlx CLI not found."
      echo "Install with: pip install vllm-mlx"
      echo "Requires Apple Silicon Mac (M1+) with macOS 14+."
      exit 1
    fi
  fi

  # Strata needs the `strata` CLI (installed via the project's installer)
  if [ "$RUNTIME" = "strata" ]; then
    if ! command -v strata >/dev/null 2>&1; then
      echo "Error: Strata CLI not found."
      echo "Install with:"
      echo "  curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh"
      exit 1
    fi
  fi
}

# Authorize with Hugging Face if token is provided
setup_huggingface() {
  if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    if command -v hf >/dev/null 2>&1; then
      echo "Logging into Hugging Face..."
      hf auth login --token "$HUGGING_FACE_HUB_TOKEN"
    else
      echo "Warning: huggingface-cli (hf) not found. Skipping HF login."
    fi
  else
    echo "Warning: HUGGING_FACE_HUB_TOKEN not set. Model download may fail if private."
  fi
}

# Clear system caches to prevent OOM (requires sudo on Linux)
clear_caches() {
  # macOS doesn't have /proc/sys/vm/drop_caches; use purge if available
  if [ "$(uname)" = "Darwin" ]; then
    if command -v purge >/dev/null 2>&1; then
      echo "Purging disk cache (macOS)..."
      purge 2>/dev/null || echo "Warning: purge failed (needs sudo). Continuing..."
    fi
    return
  fi

  # Linux: drop_caches
  if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
    echo "Clearing system caches to prevent OOM..."
    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  else
    echo "Warning: Cannot clear caches without sudo privileges. Continuing..."
  fi
}

# Stop existing container or process if running
stop_existing_container() {
  resolve_runtime
  local container_name="$1"

  if [ "$RUNTIME" = "sglang" ] || [ "$RUNTIME" = "vllm" ]; then
    # Docker-based: stop by container name
    if docker ps -q -f name="$container_name" | grep -q .; then
      echo "Stopping existing container: $container_name"
      docker stop "$container_name" >/dev/null 2>&1 || true
    fi
  elif [ "$RUNTIME" = "strata" ]; then
    # Strata manages its own server lifecycle; --restart in run_strata also
    # replaces whatever is serving, but stop explicitly for a clean log line.
    if command -v strata >/dev/null 2>&1 && strata launch --status 2>/dev/null | grep -qi "serving"; then
      echo "Stopping existing Strata server"
      strata launch --stop >/dev/null 2>&1 || true
    fi
  else
    # Native Python (freetoken, mlx): kill by process name match
    # Look for processes matching the model name on any port
    local pids
    pids=$(pgrep -f "$container_name" 2>/dev/null || true)
    if [ -n "$pids" ]; then
      echo "Stopping existing process: $container_name (pids: $pids)"
      echo "$pids" | xargs kill 2>/dev/null || true
      sleep 2
      # Force kill if still running
      pids=$(pgrep -f "$container_name" 2>/dev/null || true)
      if [ -n "$pids" ]; then
        echo "Force killing: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
      fi
    fi
  fi
}

# =============================================================================
# Dispatcher — routes to the selected runtime
# =============================================================================
# Usage: run_container <account> <model> <host_port> [additional_args...]
#
# Routes to the appropriate runtime launcher based on $RUNTIME:
#   sglang    → run_sglang_container() (Docker)
#   vllm      → run_vllm_container()   (Docker)
#   freetoken → run_freetoken()        (native Python, no Docker)
#   mlx       → run_mlx()              (native Python, no Docker, Apple Silicon)
#   strata    → run_strata()           (native CLI, no Docker, DGX Spark tuned)
#
# For TTS/STT and other non-LLM services, model scripts call run_custom_container()
# or run_native() directly instead of using this dispatcher.
run_container() {
  resolve_runtime
  case "$RUNTIME" in
    sglang)
      run_sglang_container "$@"
      ;;
    vllm)
      run_vllm_container "$@"
      ;;
    freetoken)
      run_freetoken "$@"
      ;;
    mlx)
      run_mlx "$@"
      ;;
    strata)
      run_strata "$@"
      ;;
    *)
      echo "Error: Unknown RUNTIME '$RUNTIME'. Use 'sglang', 'freetoken', 'vllm', 'mlx', or 'strata'." >&2
      exit 1
      ;;
  esac
}

# =============================================================================
# SGLang container runner
# =============================================================================
# Usage: run_sglang_container <account> <model> <host_port> [additional_args...]
run_sglang_container() {
  local account="$1"
  local model="$2"
  local host_port="$3"
  shift 3
  local additional_args=("$@")

  local local_cache_dir="${LOCAL_CACHE_DIR:-$DEFAULT_LOCAL_CACHE_DIR}"
  local container_cache_dir="${CONTAINER_CACHE_DIR:-$DEFAULT_CONTAIN_CACHE_DIR}"
  local tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-$DEFAULT_TENSOR_PARALLEL_SIZE}"
  local gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_MEMORY_UTILIZATION}"
  local otlp_endpoint="${OTLP_TRACES_ENDPOINT:-}"
  local kv_cache_dtype="${KV_CACHE_DTYPE:-$DEFAULT_KV_CACHE_DTYPE}"

  echo "Starting SGLang container for model: $account/$model"
  echo "Host Port: $host_port"
  echo "Tensor Parallel Size: $tensor_parallel_size"
  echo "Mem Fraction Static: $gpu_memory_utilization"
  if [ -n "$kv_cache_dtype" ]; then
    echo "KV Cache Dtype: $kv_cache_dtype"
  else
    echo "KV Cache Dtype: (auto/BF16)"
  fi
  echo "Image: $SGLANG_IMAGE"
  if [ -n "$otlp_endpoint" ]; then
    echo "OTLP Traces Endpoint: $otlp_endpoint"
  else
    echo "OTLP Traces Endpoint: (disabled)"
  fi

  nvidia-smi

  local -a otel_args=()
  local -a otel_env=()
  if [ -n "$otlp_endpoint" ]; then
    otel_args+=(--otlp-traces-endpoint "$otlp_endpoint")
    otel_env+=(-e "OTEL_EXPORTER_OTLP_ENDPOINT=${otlp_endpoint}")
  fi

  # FP8 KV cache: SGLang support varies by version. If KV_CACHE_DTYPE is set,
  # pass --kv-cache-dtype. If the SGLang version doesn't support it, the flag
  # will be rejected and the container will fail to start — remove the env
  # var or upgrade SGLang.
  local -a kv_args=()
  if [ -n "$kv_cache_dtype" ]; then
    kv_args+=(--kv-cache-dtype "$kv_cache_dtype")
  fi

  docker run --rm --gpus all -p "$host_port:8000" \
    --ipc=host \
    --shm-size 32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$local_cache_dir/huggingface:$container_cache_dir/huggingface" \
    -v "$local_cache_dir/sglang:$container_cache_dir/sglang" \
    -e "HF_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
    -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
    "${otel_env[@]}" \
    --name "$model" \
    "$SGLANG_IMAGE" \
    python3 -m sglang.launch_server \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --tp "$tensor_parallel_size" \
    --mem-fraction-static "$gpu_memory_utilization" \
    "${otel_args[@]}" \
    "${kv_args[@]}" \
    "${additional_args[@]}" \
    --model-path "$account/$model"
}

# =============================================================================
# vLLM container runner (fallback runtime)
# =============================================================================
# Usage: run_vllm_container <account> <model> <host_port> [additional_args...]
run_vllm_container() {
  local account="$1"
  local model="$2"
  local host_port="$3"
  shift 3
  local additional_args=("$@")

  local local_cache_dir="${LOCAL_CACHE_DIR:-$DEFAULT_LOCAL_CACHE_DIR}"
  local container_cache_dir="${CONTAINER_CACHE_DIR:-$DEFAULT_CONTAIN_CACHE_DIR}"
  local tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-$DEFAULT_TENSOR_PARALLEL_SIZE}"
  local gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_MEMORY_UTILIZATION}"
  local max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-$DEFAULT_MAX_NUM_BATCHED_TOKENS}"
  local flashinfer_moe="${VLLM_USE_FLASHINFER_MOE_FP16:-0}"
  local memory_profiler_cuda="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"
  local otlp_endpoint="${OTLP_TRACES_ENDPOINT:-}"
  local kv_cache_dtype="${KV_CACHE_DTYPE:-$DEFAULT_KV_CACHE_DTYPE}"

  echo "Starting vLLM container for model: $account/$model"
  echo "Host Port: $host_port"
  echo "Tensor Parallel Size: $tensor_parallel_size"
  echo "GPU Memory Utilization: $gpu_memory_utilization"
  echo "FlashInfer MoE FP16: $flashinfer_moe"
  echo "Memory Profiler CUDA Graphs: $memory_profiler_cuda"
  if [ -n "$kv_cache_dtype" ]; then
    echo "KV Cache Dtype: $kv_cache_dtype"
  else
    echo "KV Cache Dtype: (auto/BF16)"
  fi
  echo "Image: $VLLM_IMAGE"
  if [ -n "$otlp_endpoint" ]; then
    echo "OTLP Traces Endpoint: $otlp_endpoint"
  else
    echo "OTLP Traces Endpoint: (disabled)"
  fi

  nvidia-smi

  local -a otel_args=()
  local -a otel_env=()
  if [ -n "$otlp_endpoint" ]; then
    otel_args+=(--otlp-traces-endpoint "$otlp_endpoint")
    otel_env+=(-e "OTEL_EXPORTER_OTLP_ENDPOINT=${otlp_endpoint}")
  fi

  # FP8 KV cache: vLLM supports --kv-cache-dtype fp8 (and fp8_e4m3, fp8_e5m2).
  # This halves KV cache memory, doubling concurrent session capacity.
  # See internal-docs/guides/fp8-kv-cache.md for details.
  local -a kv_args=()
  if [ -n "$kv_cache_dtype" ]; then
    kv_args+=(--kv-cache-dtype "$kv_cache_dtype")
  fi

  docker run --rm --gpus all -p "$host_port:8000" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$local_cache_dir/huggingface:$container_cache_dir/huggingface" \
    -v "$local_cache_dir/torch:$container_cache_dir/torch" \
    -v "$local_cache_dir/torch_extensions:$container_cache_dir/torch_extensions" \
    -v "$local_cache_dir/vllm:$container_cache_dir/vllm" \
    -v "$local_cache_dir/flashinfer:$container_cache_dir/flashinfer" \
    -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
    -e "VLLM_USE_FLASHINFER_MOE_FP16=${flashinfer_moe}" \
    -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${memory_profiler_cuda}" \
    "${otel_env[@]}" \
    --name "$model" \
    "$VLLM_IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size "$tensor_parallel_size" \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    "${otel_args[@]}" \
    "${kv_args[@]}" \
    "${additional_args[@]}" \
    --model "$account/$model"
}

# =============================================================================
# FreeToken runner (native Python, no Docker — MoE default per ADR amendment)
# =============================================================================
# Usage: run_freetoken <account> <model> <host_port> [additional_args...]
#
# FreeToken is the default runtime for MoE models (both in-VRAM and frontier)
# per ADR-202608021744 as amended 2026-08-26. It runs as a native Python
# process — no Docker container. The `ft` CLI must be installed:
#
#   uv pip install "freetoken[accel]"
#
# FreeToken supports MXFP4, NVFP4, FP8, BF16, and GGUF (Gemma-4 only).
# It does NOT support GPTQ — use RUNTIME=vllm for GPTQ models.
#
# FreeToken's default port is 1919; we override with --port to stay within
# the repo's 8000+ port convention.
#
# Environment variables:
#   FREETOKEN_MOE_BACKEND  — auto|fused|offload|cpu|hybrid (default: auto)
#   FREETOKEN_BENCH_BW     — if "1", run `ft bench bw` before serving
#
run_freetoken() {
  local account="$1"
  local model="$2"
  local host_port="$3"
  shift 3
  local additional_args=("$@")

  local moe_backend="${FREETOKEN_MOE_BACKEND:-auto}"
  local run_bench="${FREETOKEN_BENCH_BW:-0}"

  echo "Starting FreeToken server for model: $account/$model"
  echo "Host Port: $host_port"
  echo "MoE Backend: $moe_backend"
  if [ -n "$FREETOKEN_VERSION" ]; then
    echo "FreeToken Version: $FREETOKEN_VERSION"
  else
    echo "FreeToken Version: (unpinned — set FREETOKEN_VERSION for reproducibility)"
  fi

  # Optional: calibrate bandwidth profile (run once per machine)
  if [ "$run_bench" = "1" ]; then
    echo "Running ft bench bw to calibrate bandwidth profile..."
    ft bench bw
  fi

  # FreeToken accepts HF repo ids or local paths via --model
  # It auto-detects dtype, attention/MoE backends, cache sizes, and
  # tool-call/reasoning parsers from the checkpoint.
  ft serve \
    --model "$account/$model" \
    --port "$host_port" \
    --moe-backend "$moe_backend" \
    "${additional_args[@]}"
}

# =============================================================================
# Strata runner (native CLI, no Docker — DGX Spark / consumer-GPU tuned engine)
# =============================================================================
# Usage: run_strata <account> <model> <host_port> [additional_args...]
#
# Strata (github.com/Niko1221/Strata) is a llama.cpp-lineage engine with a
# first-class DGX Spark port: linux-aarch64 engine, sm_121 perf defaults,
# --mmap-experts for the 128 GB unified memory pool, and measured ~46 tok/s
# decode on Qwen3.8-Flash-Next (see docs/DGX_SPARK_LINUX.md upstream).
# It serves OpenAI- and Anthropic-compatible APIs.
#
# Install (no sudo; lands in ~/.local/share/strata, `strata` on PATH):
#   curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
#
# Two model modes:
#   Catalog pack — set STRATA_FAMILY + STRATA_PACK for a tuned catalog entry
#     (e.g. STRATA_FAMILY=nvfp4 STRATA_PACK=nvfp4, or family qwen pack iq3_s).
#     `strata launch` downloads the pack itself.
#   HF GGUF repo — leave STRATA_FAMILY unset; $ACCT/$MODEL is treated as a
#     Hugging Face repo, `hf download` fetches STRATA_GGUF_PATTERN-matching
#     files into $STRATA_DATA_DIR/models/<model>/, and the folder is passed
#     via --model-path. Use this for abliterated/community GGUF quants.
#
# NOTE: `strata launch` with --no-tui starts the server as a managed
# background process and returns — this function returns while the model
# keeps serving (unlike the Docker runners, which stay foregrounded).
#
# Environment variables:
#   STRATA_FAMILY        — catalog model family (e.g. qwen, nvfp4); unset = HF repo mode
#   STRATA_PACK          — catalog pack name (default: $MODEL)
#   STRATA_GGUF_PATTERN  — hf download --include pattern for HF repo mode
#                          (default: "*", i.e. all files — set it!)
#   STRATA_DATA_DIR      — Strata data/model dir (default: $LOCAL_CACHE_DIR/strata)
#   STRATA_CONTEXT       — context size (default: 64k)
#   STRATA_ACCESS        — local|lan|cloudflare|tailscale (default: lan, to
#                          match the Docker runners' 0.0.0.0 binding)
#   STRATA_API_KEY       — if set, clients must send this key; unset = --no-api-key
#
run_strata() {
  local account="$1"
  local model="$2"
  local host_port="$3"
  shift 3
  local additional_args=("$@")

  local local_cache_dir="${LOCAL_CACHE_DIR:-$DEFAULT_LOCAL_CACHE_DIR}"
  local data_dir="${STRATA_DATA_DIR:-$local_cache_dir/strata}"
  local context="${STRATA_CONTEXT:-64k}"
  local access="${STRATA_ACCESS:-lan}"

  local -a launch_args=(
    --context "$context"
    --access "$access"
    --port "$host_port"
    --data-dir "$data_dir"
    --no-tui
    --yes
    --restart
  )

  # Repo convention is unauthenticated LAN endpoints; match it by default.
  if [ -n "${STRATA_API_KEY:-}" ]; then
    launch_args+=(--api-key "$STRATA_API_KEY")
  else
    launch_args+=(--no-api-key)
  fi

  if [ -n "${STRATA_FAMILY:-}" ]; then
    launch_args+=(--family "$STRATA_FAMILY" --model "${STRATA_PACK:-$model}")
    echo "Starting Strata server (catalog pack): $STRATA_FAMILY/${STRATA_PACK:-$model}"
  else
    # HF repo mode: download GGUF files and serve the folder via --model-path
    local model_dir="$data_dir/models/$model"
    echo "Starting Strata server (HF GGUF): $account/$model"
    if ! command -v hf >/dev/null 2>&1; then
      echo "Error: 'hf' CLI required to download $account/$model." >&2
      echo "Install with: uv tool install huggingface-hub" >&2
      exit 1
    fi
    if ! compgen -G "$model_dir/*.gguf" >/dev/null 2>&1 && [ -z "$(find "$model_dir" -name '*.gguf' -print -quit 2>/dev/null)" ]; then
      echo "Downloading $account/$model (pattern: ${STRATA_GGUF_PATTERN:-*}) -> $model_dir"
      mkdir -p "$model_dir"
      hf download "$account/$model" \
        --include "${STRATA_GGUF_PATTERN:-*}" \
        --local-dir "$model_dir"
    else
      echo "Using cached model files in $model_dir"
    fi
    # --model-path wants "a folder of shards or one .gguf" — hf download may
    # nest the shards in a subdir (e.g. swift-IQ3_S/), so point at the dir
    # that actually contains the .gguf files.
    local model_path
    model_path="$(dirname "$(find "$model_dir" -name '*.gguf' -print -quit)")"
    if [ -z "$model_path" ] || [ ! -d "$model_path" ]; then
      echo "Error: no .gguf files found under $model_dir after download." >&2
      exit 1
    fi
    launch_args+=(--model-path "$model_path")
  fi

  echo "Host Port: $host_port"
  echo "Context: $context"
  echo "Access: $access"
  echo "Data Dir: $data_dir"
  if [ -n "$STRATA_VERSION" ]; then
    echo "Strata Version: $STRATA_VERSION"
  else
    echo "Strata Version: (unpinned — set STRATA_VERSION for reproducibility)"
  fi

  mkdir -p "$data_dir"
  strata launch "${launch_args[@]}" "${additional_args[@]}"
}

# =============================================================================
# MLX runner (native Python, no Docker — Apple Silicon Mac serving)
# =============================================================================
# Usage: run_mlx <account> <model> <host_port> [additional_args...]
#
# MLX is the runtime for Apple Silicon Mac hosts (Mac Studio, MacBook Pro,
# Mac Mini with M1+ chips). It uses Apple's MLX framework via vllm-mlx,
# which provides vLLM-style serving with:
#   - Continuous batching
#   - Paged KV cache
#   - Prefix caching
#   - SSD-tiered cache
#   - OpenAI + Anthropic API compatibility
#   - Multimodal model support
#
# Install:
#   pip install vllm-mlx
#
# Requires Apple Silicon (M1+) with macOS 14+. Unified memory is used
# directly — no conversion step. Models from mlx-community on HuggingFace
# are pre-converted to MLX format; other safetensors checkpoints are
# auto-converted on first load.
#
# Environment variables:
#   MLX_CONTINUOUS_BATCHING — if "1", enable continuous batching (default: 1)
#   MLX_MAX_KV_SIZE         — max KV cache size in tokens (optional)
#
run_mlx() {
  local account="$1"
  local model="$2"
  local host_port="$3"
  shift 3
  local additional_args=("$@")

  local continuous_batching="${MLX_CONTINUOUS_BATCHING:-1}"

  echo "Starting vllm-mlx server for model: $account/$model"
  echo "Host Port: $host_port"
  echo "Continuous Batching: $continuous_batching"
  if [ -n "$MLX_VERSION" ]; then
    echo "vllm-mlx Version: $MLX_VERSION"
  else
    echo "vllm-mlx Version: (unpinned — set MLX_VERSION for reproducibility)"
  fi

  # vllm-mlx serves on the specified port with OpenAI + Anthropic APIs
  # The --continuous-batching flag enables vLLM-style batched serving
  local -a mlx_args=()
  if [ "$continuous_batching" = "1" ]; then
    mlx_args+=(--continuous-batching)
  fi

  vllm-mlx serve \
    --port "$host_port" \
    "${mlx_args[@]}" \
    "${additional_args[@]}" \
    "$account/$model"
}

# =============================================================================
# Custom container runner (for non-LLM services: TTS, STT, etc.)
# =============================================================================
# Usage: run_custom_container <image> <container_name> <host_port> <cmd...>
#
# Runs a Docker container with GPU passthrough, cache volumes, and port
# mapping — but with a caller-specified image and command. Used by
# model-tts.sh and model-stt.sh for purpose-built Docker images that
# don't fit the LLM runtime dispatch (SGLang/vLLM/FreeToken/MLX).
#
# The function provides the common Docker boilerplate:
#   --gpus all              GPU passthrough (NVIDIA)
#   --ipc=host              Shared memory for efficient GPU communication
#   --shm-size 8g           Shared memory for model loading
#   HF cache volumes        HuggingFace model cache mounted from host
#   HF token env vars       HUGGING_FACE_HUB_TOKEN passed through
#
# The caller specifies:
#   $1 — Docker image (e.g. "hwdsl2/kokoro-server:cuda")
#   $2 — Container name (used for stop_existing_container)
#   $3 — Host port (mapped to container's internal port via $CUSTOM_CONTAINER_PORT)
#   $@ — Remaining args are the container's command (optional; most images
#        have a default ENTRYPOINT/CMD)
#
# Environment variables:
#   CUSTOM_CONTAINER_PORT  — internal container port (default: 8880 for TTS,
#                            9000 for STT; set in the model script)
#   CUSTOM_IMAGE           — override the image from the model script
#   LOCAL_CACHE_DIR        — host cache directory (same as LLM runners)
#   CONTAINER_CACHE_DIR    — container cache directory (same as LLM runners)
#
# On Apple Silicon, this function falls back to running without --gpus
# (Docker on Mac doesn't support GPU passthrough; TTS/STT models are small
# enough to run on CPU or via a native Python process instead).
#
run_custom_container() {
  local image="$1"
  local container_name="$2"
  local host_port="$3"
  shift 3
  local cmd=("$@")

  local local_cache_dir="${LOCAL_CACHE_DIR:-$DEFAULT_LOCAL_CACHE_DIR}"
  local container_cache_dir="${CONTAINER_CACHE_DIR:-$DEFAULT_CONTAIN_CACHE_DIR}"
  local internal_port="${CUSTOM_CONTAINER_PORT:-8880}"

  # Allow image override via env var (useful for testing alternative images)
  image="${CUSTOM_IMAGE:-$image}"

  echo "Starting custom container: $container_name"
  echo "Image: $image"
  echo "Host Port: $host_port -> $internal_port"

  local -a gpu_args=()
  if [ "$(detect_system)" = "nvidia" ]; then
    gpu_args+=(--gpus all)
    nvidia-smi
  else
    echo "Note: No NVIDIA GPU detected. Running without GPU acceleration."
    echo "TTS/STT models are small enough for CPU inference on most hardware."
  fi

  # Build the docker run command with common boilerplate.
  # The HF cache volume lets the container reuse downloaded models
  # instead of re-downloading on every start.
  # The --shm-size is smaller than LLM runners (8g vs 32g) because
  # TTS/STT models are much smaller and don't need as much shared memory.
  docker run --rm \
    "${gpu_args[@]}" \
    -p "$host_port:$internal_port" \
    --ipc=host \
    --shm-size 8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$local_cache_dir/huggingface:$container_cache_dir/huggingface" \
    -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
    -e "HF_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
    --name "$container_name" \
    "$image" \
    "${cmd[@]}"
}

# =============================================================================
# Native process runner (for TTS/STT on Apple Silicon or CPU-only systems)
# =============================================================================
# Usage: run_native <container_name> <host_port> <cmd...>
#
# Runs a native Python process (no Docker) for TTS/STT on systems where
# Docker GPU passthrough isn't available (Apple Silicon, CPU-only hosts).
# The caller provides the command to run; this function handles process
# lifecycle (stop existing, start, wait).
#
# On Apple Silicon, TTS/STT models run via:
#   - MPS (Metal Performance Shaders) for GPU acceleration, or
#   - CPU (models are small enough for real-time CPU inference)
#
# The caller is responsible for installing the Python package and
# providing the correct command. This function just manages the process.
#
run_native() {
  local container_name="$1"
  local host_port="$2"
  shift 2
  local cmd=("$@")

  echo "Starting native process: $container_name"
  echo "Port: $host_port"

  # Run the command directly. The stop_existing_container() function
  # (called by the model script before this) handles killing any
  # previous instance.
  exec "${cmd[@]}"
}
