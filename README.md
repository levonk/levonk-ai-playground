# vLLM Model Runner Scripts

Primarily tracks top performers from https://github.com/AlexsJones/llmfit for Nvidia GB10 128GB, CUDA w/ vLLM

## Architecture

This project uses a **shared library pattern** for DRY (Don't Repeat Yourself) code organization:

- **`vllm-runner-lib.sh`** - Shared library with common functions
- **`model-*.sh`** - Model-specific runner scripts that source the library

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

| Script | Model | Account | Host Port | Special Config |
|--------|-------|---------|-----------|----------------|
| `model-chat.sh` | Qwen3-Next-80B-A3B-Instruct-AWQ-4bit | cyankiwi | 8000 | `--quantization awq_merlin` |
| `model-code.sh` | Qwen3-Coder-Next-AWQ-4bit | cyankiwi | 8001 | `--quantization awq_merlin` |
| `model-reasoning.sh` | Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled | hesamation | 8002 | `--quantization awq_merlin` |
| `model-general.sh` | Qwen3-Next-80B-A3B-Thinking-AWQ-4bit | cyankiwi | 8003 | `--max-model-len 262144`, speculative decoding |

## Usage

### Basic Usage

Run a specific model:

```bash
./model-chat.sh
./model-code.sh
./model-reasoning.sh
./model-general.sh
```

### Custom Configuration

Override configuration via environment variables:

```bash
# Change port
HOST_PORT=9000 ./model-chat.sh

# Change model
MODEL=custom-model-name ./model-chat.sh

# Change account
ACCT=my-account ./model-chat.sh

# GPU memory utilization
GPU_MEMORY_UTILIZATION=0.95 ./model-chat.sh

# Tensor parallel size (for multi-GPU)
TENSOR_PARALLEL_SIZE=2 ./model-chat.sh
```

### Running Multiple Models Simultaneously

Since each model uses a unique port, you can run them simultaneously:

```bash
# Terminal 1
./model-chat.sh

# Terminal 2
./model-code.sh

# Terminal 3
./model-reasoning.sh
```

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
- `HUGGING_FACE_HUB_TOKEN` - HF token for private models (loaded from `.env`)

## Adding a New Model

Create a new script following this pattern:

```bash
#!/usr/bin/env bash
#
# vLLM Your Model Runner
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./vllm-runner-lib.sh
source "$SCRIPT_DIR/vllm-runner-lib.sh"

# Model-specific configuration
ACCT="${ACCT:-your-account}"
MODEL="${MODEL:-your-model-name}"
HOST_PORT="${HOST_PORT:-8004}"  # Use a unique port

main() {
  load_env
  check_prerequisites
  setup_huggingface
  clear_caches
  stop_existing_container "$MODEL"

  # Model-specific vLLM arguments
  run_vllm_container "$ACCT" "$MODEL" "$HOST_PORT" \
    --your-custom-flag value
}

main "$@"
```

## Shared Library Functions

The `vllm-runner-lib.sh` provides:

- `load_env()` - Load `.env` file if present
- `check_prerequisites()` - Verify Docker and sudo availability
- `setup_huggingface()` - Authenticate with Hugging Face
- `clear_caches()` - Clear system caches (requires sudo)
- `stop_existing_container()` - Stop running container if exists
- `run_vllm_container()` - Run vLLM container with standard configuration
