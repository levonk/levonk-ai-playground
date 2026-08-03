---
date:
  created: "2026-08-01"
  knowledge-basis: "2026-08-01"
  last-used: "2026-08-01"
---

# Developer Guide: levonk-ai-playground

This guide is for developers editing the scripts. For user-facing project
overview and deploy instructions, see the root
[`AGENTS.md`](../../AGENTS.md).

## JIT Index

- Out of Scope: [`internal-docs/oos/`](../../internal-docs/oos/) — what this repo explicitly does NOT do (check before adding features)
- Improvements: [`internal-docs/improvements/INDEX.md`](../../internal-docs/improvements/INDEX.md) — potential improvements to consider (check before proposing changes)
- Anti-Patterns: [`internal-docs/anti-patterns/INDEX.md`](../../internal-docs/anti-patterns/INDEX.md) — things explicitly NOT to do (check before implementing changes)

## Setup (Development Environment)

This repo has no build, test, or lint pipeline — it is a collection of
deployment shell scripts. "Setup" means getting the runtime prerequisites on
the GPU host.

**Prerequisites:**
- NVIDIA GB10 128GB with CUDA and NVIDIA Container Toolkit
- Docker (`docker info` must succeed)
- `sudo` (cache clearing writes to `/proc/sys/vm/drop_caches`)
- `jq` (optional — test-query scripts fall back to `sed`)
- A Hugging Face token in `.env` as `HUGGING_FACE_HUB_TOKEN` (required for private/gated models)

**devbox:** A `devbox.json` is present and declares `git`, `just`,
`python315`, `huggingface-hub`, `ray`, `llmfit`, rtk bundles, `smartfo`, and
`docker`. Enter the environment with `devbox shell`. Note: `devbox.json`
references `just` targets (`bootstrap-internal`, `doctor-internal`, etc.) but
**no `justfile` exists in this repo** — those targets are inherited from the
rtk bundles. Do not assume `just build` / `just test` work here; verify before
relying on them.

## Commands

There are no build/test/lint commands. The commands that exist are the scripts
themselves.

**Deploy a model:**
```bash
./model-chat.sh       # Qwen3-Next-80B-A3B-Instruct-AWQ-4bit  -> host port 8000
./model-code.sh       # Qwen3-Coder-Next-AWQ-4bit             -> host port 8001
./model-reason.sh     # Qwen3.5-35B reasoning-distilled        -> host port 8002
./model-general.sh    # Qwen3-Next-80B-A3B-Thinking-AWQ-4bit  -> host port 8003
```

**Smoke-test a running server:**
```bash
./test-query.sh                       # auto-discovers running LLM Docker containers
./test-query.sh 'hello world'         # custom prompt, all containers
./test-query.sh -p 8002 -m qwen3 'hi' # single port, explicit model
./test-query-local.sh                 # targets remote host `ai` port 8002
```

**Override a model script via env vars** (see `vllm-runner-lib.sh` defaults):
```bash
HOST_PORT=9000 ./model-chat.sh
MODEL=custom-model ACCT=my-account ./model-chat.sh
GPU_MEMORY_UTILIZATION=0.95 TENSOR_PARALLEL_SIZE=2 ./model-chat.sh
OTLP_TRACES_ENDPOINT=http://openlit:4318 ./model-chat.sh   # enable OTLP tracing
```

## Tech Stack / Environment

- **Bash** — all scripts use `set -euo pipefail`
- **Docker** — `docker run --rm --gpus all` against `nvcr.io/nvidia/vllm:26.04-py3`
- **vLLM** — OpenAI-compatible server (`vllm.entrypoints.openai.api_server`)
- **Hugging Face Hub** — `hf auth login --token` for gated/private models
- **devbox (Nix)** — reproducible environment declared in `devbox.json`
- **OpenTelemetry (optional)** — `OTLP_TRACES_ENDPOINT` enables `--otlp-traces-endpoint` in vLLM and LiteLLM distributed tracing

## Workflow

1. Branch from `main`: `feature/{name}`, `fix/{name}`, or `chore/{name}`
2. Edit the script(s); follow the shared-library pattern (see Patterns)
3. Smoke-test on the GPU box: run the affected `model-*.sh`, then `./test-query.sh`
4. Commit with a conventional message (`feat(model):`, `fix(lib):`, `chore(model):`)
5. Rebase on `main` if diverged
6. Open a PR

There is no CI and no automated test suite. Verification is manual: run the
model script and confirm the vLLM server responds via `test-query.sh`.

## Key Directories

- `./` (root) — all model runner scripts and the shared library live flat at the repo root
- `.agents/workflow/` — workflow pointers (e.g., the git-repository-management skill trigger)
- `.agents/knowledge/` — this developer guide
- `.devbox/` — devbox-generated artifacts (gitignored, do not commit)
- `internal-docs/` — out-of-scope, improvements, anti-patterns docs

## Key Files

**Scripts:**
- `vllm-runner-lib.sh` — shared library; every `model-*.sh` sources this. Provides `load_env`, `check_prerequisites`, `setup_huggingface`, `clear_caches`, `stop_existing_container`, `run_vllm_container`
- `model-*.sh` — one runner per model. Sets `ACCT`, `MODEL`, `HOST_PORT`, optional vLLM flags, then calls `run_vllm_container`
- `test-query.sh` — auto-discovers Docker LLM containers and POSTs an OpenAI-style chat completion to each
- `test-query-local.sh` — targets a single remote OpenAI-compatible endpoint (no Docker discovery)

**Config:**
- `devbox.json` — Nix/devbox environment declaration
- `.env` — gitignored; holds `HUGGING_FACE_HUB_TOKEN` and optional `OTLP_TRACES_ENDPOINT`
- `.gitignore` — ignores `.env`, `*.log`, `*.bak`, `*.swp`, `.devbox/`

**Docs:**
- `README.md` — architecture rationale, model table, env-var reference, "add a new model" template
- `AGENTS.md` — agent guidance (primary)
- `CLAUDE.md` / `AGENT.md` — referrals to `AGENTS.md`

## Patterns

**Shared library pattern (DRY):**
- ✅ DO put shared Docker/vLLM logic in `vllm-runner-lib.sh`
- ✅ DO source the library at the top of every `model-*.sh`: `source "$SCRIPT_DIR/vllm-runner-lib.sh"`
- ✅ DO keep each `model-*.sh` thin: set `ACCT`/`MODEL`/`HOST_PORT` and model-specific flags, then call `run_vllm_container`
- ✅ DO use a unique `HOST_PORT` per model so models run simultaneously
- ✅ DO use `set -euo pipefail` in every script
- ❌ DON'T duplicate Docker run flags across model scripts — extend the library instead
- ❌ DON'T hardcode a host port that another `model-*.sh` already uses

**Adding a new model** (template from README):
```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./vllm-runner-lib.sh
source "$SCRIPT_DIR/vllm-runner-lib.sh"
ACCT="${ACCT:-your-account}"
MODEL="${MODEL:-your-model-name}"
HOST_PORT="${HOST_PORT:-8004}"  # pick a free port
main() {
  load_env; check_prerequisites; setup_huggingface
  clear_caches; stop_existing_container "$MODEL"
  run_vllm_container "$ACCT" "$MODEL" "$HOST_PORT" --your-flag value
}
main "$@"
```

**Env-var overrides:** All tunable values (`HOST_PORT`, `MODEL`, `ACCT`,
`GPU_MEMORY_UTILIZATION`, `TENSOR_PARALLEL_SIZE`, `MAX_NUM_BATCHED_TOKENS`,
`LOCAL_CACHE_DIR`, `CONTAINER_CACHE_DIR`, `OTLP_TRACES_ENDPOINT`) read from
env with defaults in `vllm-runner-lib.sh`. Keep this pattern for new knobs.

## Boundaries

### Always
- Source `vllm-runner-lib.sh` from every `model-*.sh`
- Use `set -euo pipefail` in every script
- Pick a unique `HOST_PORT` for any new model script
- Keep the shared library the single source of Docker/vLLM run logic
- Smoke-test with `test-query.sh` after changing a runner or the library

### Ask First
- Changing the pinned vLLM image (`nvcr.io/nvidia/vllm:26.04-py3`) — affects every model
- Changing default `GPU_MEMORY_UTILIZATION` or `TENSOR_PARALLEL_SIZE` in the library — affects every model
- Adding a host port below 8000 or above 9999 (outside the current 8000–8003 convention)
- Switching the base model for an existing script (see Anti-Patterns — the Intel AutoRound revert)

### Never
- Commit `.env` or any `HUGGING_FACE_HUB_TOKEN` value
- Hardcode a HF token in a script
- Run `docker run` without `--rm` (the library always uses `--rm`; leaving containers around breaks `stop_existing_container` assumptions)
- Assume flags tuned for GB10 generalize to other GPUs without re-tuning
- Bypass `vllm-runner-lib.sh` and inline Docker run logic in a model script

## Known Gotchas

- **No justfile**: `devbox.json` declares `just` and references `just *-internal` targets, but there is no `justfile` in this repo. The targets come from rtk bundles. Verify they resolve before relying on `just build`/`just test`.
- **`clear_caches` needs sudo**: it writes `3` to `/proc/sys/vm/drop_caches`. On hosts without passwordless sudo it warns and continues — not a hard failure.
- **`hf` vs `huggingface-cli`**: `setup_huggingface` calls `hf auth login`. If only `huggingface-cli` is installed, HF login is skipped with a warning.
- **Container name = model name**: `run_vllm_container` names the container `$MODEL`. `stop_existing_container "$MODEL"` relies on this. If you override `MODEL` via env, the stop logic still works because both use the same `$MODEL`.
- **OTLP is opt-in**: `OTLP_TRACES_ENDPOINT` empty/unset = tracing disabled. Only set it when an OTLP collector (e.g., OpenLit) is reachable.
- **Speculative decoding is model-specific**: `model-general.sh` passes `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`. This only works for Qwen3-Next MTP models — do not copy it to other model scripts.
- **README model table can drift from scripts**: the README lists `model-reasoning.sh` but the actual file is `model-reason.sh`. Trust the scripts, not the table, when they disagree.

## Definition of Done

A change to a runner or the library is done when:

- [ ] The affected `model-*.sh` runs on the GPU box without manual intervention
- [ ] `./test-query.sh` returns HTTP 200 from the deployed container
- [ ] No new host port collides with an existing `model-*.sh`
- [ ] Shared logic was extended in `vllm-runner-lib.sh`, not duplicated in the runner
- [ ] `.env` and any token values are absent from the diff
- [ ] The README model table and env-var list still match the scripts (or are updated)
- [ ] The closest `AGENTS.md` chain was re-read and any stale text removed
