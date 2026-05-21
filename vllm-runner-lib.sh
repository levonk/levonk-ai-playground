#!/usr/bin/env bash
#
# vLLM Model Runner Library
# Shared functions for deploying vLLM models with Docker
#

set -euo pipefail

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

# Run a vLLM container
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

  echo "Starting vLLM container for model: $account/$model"
  echo "Host Port: $host_port"
  echo "Tensor Parallel Size: $tensor_parallel_size"
  echo "GPU Memory Utilization: $gpu_memory_utilization"

  docker run -it --rm --gpus all -p "$host_port:8000" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$local_cache_dir/huggingface:$container_cache_dir/huggingface" \
    -v "$local_cache_dir/torch:$container_cache_dir/torch" \
    -v "$local_cache_dir/torch_extensions:$container_cache_dir/torch_extensions" \
    -v "$local_cache_dir/vllm:$container_cache_dir/vllm" \
    -v "$local_cache_dir/flashinfer:$container_cache_dir/flashinfer" \
    -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
    --name "$model" \
    nvcr.io/nvidia/vllm:26.04-py3 \
    python3 -m vllm.entrypoints.openai.api_server \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size "$tensor_parallel_size" \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    "${additional_args[@]}" \
    --model "$account/$model"
}
