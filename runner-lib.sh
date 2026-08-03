#!/usr/bin/env bash
#
# LLM Model Runner Library
# Shared functions for deploying LLM models with Docker
#
# Supports two inference runtimes:
#   - SGLang (cu130) — default for supported in-VRAM models on GB10/Blackwell
#   - vLLM           — fallback for models/quants/parallelism SGLang does not support
#
# See ADR-202608021744 for the tiered runtime selection rationale.
#

set -euo pipefail

# =============================================================================
# Runtime Selection — change this to swap the inference engine
# =============================================================================
# Set RUNTIME to "sglang" or "vllm".  All scripts that source this library
# and call run_container() will use the selected runtime.
#
# Individual scripts can override by exporting RUNTIME before sourcing this
# file, e.g.:
#
#     RUNTIME=vllm source "$SCRIPT_DIR/runner-lib.sh"
#
# Per ADR-202608021744: SGLang is the default; vLLM is retained as fallback
# for GPTQ, compressed-tensors, multi-GPU pipeline parallel, and models
# SGLang does not yet support.
#
# NOTE: Model-specific args passed to run_container() (e.g. --max-model-len,
# --speculative-config) are runtime-specific.  When switching runtimes you
# may need to adjust those args in the individual model-*.sh scripts.
# =============================================================================
RUNTIME="${RUNTIME:-sglang}"

# Pinned image tags (do NOT use floating "latest" tags in production)
SGLANG_IMAGE="lmsysorg/sglang:v0.5.16-cu130"
VLLM_IMAGE="nvcr.io/nvidia/vllm:26.04-py3"

# Default configuration
DEFAULT_LOCAL_CACHE_DIR="/root/.cache"
DEFAULT_CONTAIN_CACHE_DIR="/root/.cache"
DEFAULT_TENSOR_PARALLEL_SIZE=1
DEFAULT_GPU_MEMORY_UTILIZATION=0.90
DEFAULT_MAX_NUM_BATCHED_TOKENS=32768

# Load environment variables from .env if it exists
load_env() {
  if [ -f .env ]; then
    # shellcheck source=/dev/null
    source .env
  fi
}

# Check prerequisites
check_prerequisites() {
  local missing=()

  command -v docker >/dev/null 2>&1 || missing+=("docker")
  command -v sudo >/dev/null 2>&1 || missing+=("sudo")

  if [ ${#missing[@]} -gt 0 ]; then
    echo "Error: Missing required commands: ${missing[*]}"
    exit 1
  fi

  # Check if Docker is running
  if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
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

# Clear system caches to prevent OOM (requires sudo)
clear_caches() {
  if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
    echo "Clearing system caches to prevent OOM..."
    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  else
    echo "Warning: Cannot clear caches without sudo privileges. Continuing..."
  fi
}

# Stop existing container if running
stop_existing_container() {
  local container_name="$1"
  if docker ps -q -f name="$container_name" | grep -q .; then
    echo "Stopping existing container: $container_name"
    docker stop "$container_name" >/dev/null 2>&1 || true
  fi
}

# =============================================================================
# Dispatcher — routes to the selected runtime
# =============================================================================
# Usage: run_container <account> <model> <host_port> [additional_args...]
run_container() {
  if [ "$RUNTIME" = "sglang" ]; then
    run_sglang_container "$@"
  elif [ "$RUNTIME" = "vllm" ]; then
    run_vllm_container "$@"
  else
    echo "Error: Unknown RUNTIME '$RUNTIME'. Use 'sglang' or 'vllm'." >&2
    exit 1
  fi
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

  echo "Starting SGLang container for model: $account/$model"
  echo "Host Port: $host_port"
  echo "Tensor Parallel Size: $tensor_parallel_size"
  echo "Mem Fraction Static: $gpu_memory_utilization"
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

  echo "Starting vLLM container for model: $account/$model"
  echo "Host Port: $host_port"
  echo "Tensor Parallel Size: $tensor_parallel_size"
  echo "GPU Memory Utilization: $gpu_memory_utilization"
  echo "FlashInfer MoE FP16: $flashinfer_moe"
  echo "Memory Profiler CUDA Graphs: $memory_profiler_cuda"
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
    "${additional_args[@]}" \
    --model "$account/$model"
}
