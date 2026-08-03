---
date:
  created: "2026-08-01"
  knowledge-basis: "2026-08-01"
  last-used: "2026-08-01"
---

# vLLM Model Runner Scripts

> Repo: `levonk-ai-playground` — https://github.com/levonk/levonk-ai-playground

## Project Snapshot

| Field | Value |
|-------|-------|
| Project name | levonk-ai-playground (vLLM Model Runner Scripts) |
| Type | Flat single-project repo (not a monorepo) |
| Category | Shell-script vLLM model deployment playground |
| Language | Bash |
| Runtime | Docker containers on NVIDIA GB10 128GB, CUDA |
| Repository | https://github.com/levonk/levonk-ai-playground |
| README | [`README.md`](README.md) — architecture rationale, model table, env-var reference |

## Project Overview

A playground for deploying vLLM-served LLMs in Docker containers on a single
NVIDIA GB10 128GB machine. The repo tracks top-performing models from
[llmfit](https://github.com/AlexsJones/llmfit) and provides one script per
model plus a shared library. Each model binds a unique host port so multiple
models run simultaneously. The audience is the operator running the GPU box.

## Install

This project has no package-manager install step. An operator deploys it by
cloning onto the GPU host and running a model script.

**Prerequisites on the GPU host:**
- NVIDIA GB10 (128 GB) with CUDA
- Docker with NVIDIA Container Toolkit (`--gpus all` works)
- `sudo` (cache clearing uses `/proc/sys/vm/drop_caches`)
- `jq` (optional — test scripts fall back to `sed`)

**Deploy:**
```bash
git clone git@github.com:levonk/levonk-ai-playground.git
cd levonk-ai-playground
printf 'HUGGING_FACE_HUB_TOKEN=hf_xxx\n' > .env   # required for private models
./model-chat.sh                                    # starts vLLM on host port 8000
```

Each `model-*.sh` pulls `nvcr.io/nvidia/vllm:26.04-py3` and starts the
OpenAI-compatible vLLM server on the script's host port. Smoke-test a running
server with `./test-query.sh` (auto-discovers Docker LLM containers) or
`./test-query-local.sh` (targets a remote host).

## JIT Index

This is a flat repo. There are no sub-package `AGENTS.md` files. The developer
guide is the single deeper doc.

- Developer Guide: [`.agents/knowledge/developer.md`](.agents/knowledge/developer.md) — Setup, commands, environment, workflow, repo structure, patterns, boundaries, gotchas, and Definition of Done for anyone editing the scripts
- Workflow: [`.agents/workflow/aiplayground-git.md`](.agents/workflow/aiplayground-git.md) — Points at the git-repository-management skill for this repo

## Knowledge Bundles

No offline bundles are installed in `.agents/knowledge/bundles/`. Run the
`project-adopter` skill's `install-knowledge-bundles.py` to populate them.
The URL-referenced bundles most relevant to this repo:

| Bundle | Read when working on… |
|--------|-----------------------|
| [container-best-practices](https://github.com/levonk/skills-releases/knowledge/container-best-practices) | Docker run flags, image pinning, GPU passthrough |
| [ai-primitives](https://github.com/levonk/skills-releases/knowledge/ai-primitives) | vLLM flags, model quantization, serving config |

## Out of Scope

For what this repo explicitly does NOT do, see
[`internal-docs/oos/`](internal-docs/oos/).

## Improvements

For potential improvements to architecture, standards, and processes, see
[`internal-docs/improvements/INDEX.md`](internal-docs/improvements/INDEX.md).
These are suggestions to consider — not decisions yet. Check before proposing
changes to avoid re-proposing already-evaluated improvements.

## Anti-Patterns

For things explicitly NOT to do (practices found harmful or inferior), see
[`internal-docs/anti-patterns/INDEX.md`](internal-docs/anti-patterns/INDEX.md).
These are negative findings — do NOT implement any approach listed there.

## Universal Contracts

- **License**: Treat as private/internal unless a LICENSE file is added.
- **Secrets**: Never commit `.env` or any `HUGGING_FACE_HUB_TOKEN` value. `.env` is gitignored.
- **Hardware target**: Scripts assume NVIDIA GB10 128GB. Flags (GPU memory utilization, tensor parallel size, quantization) are tuned for that card. Do not assume they generalize to other GPUs without re-tuning.

## Agent Interaction Protocol

Anytime you have a question for the user — mid-task, at a decision point, or
when ambiguity blocks progress — present it as **question + recommendation +
why**, in that order. Do not ask a bare question and wait. The user should be
able to reply with a single letter, a "yes/no", or "go ahead" without typing
out the reasoning.

For each question: (1) the question in one plain sentence, (2) the option you
would pick labeled `(recommended)`, (3) one or two sentences on the trade-off
(name what breaks or what is lost if the user picks the other option).

Do not ask when the answer is already clear from the prompt, the codebase, or
prior context — proceed and state your assumption. Do not ask about reversible
low-stakes decisions — pick the default, note it, and move on.

For high-stakes trade-offs (architecture, destructive actions, one-way doors)
or before generating/updating an artifact, escalate to the full
clarifying-questions protocol.

## Developer Guide

For workflows, repository structure, code style, boundaries, known gotchas,
and the Definition of Done checklist, see
[`.agents/knowledge/developer.md`](.agents/knowledge/developer.md).
