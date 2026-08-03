# 🛑 Anti-Pattern: Intel AutoRound Qwen3-Coder-Next-int4 on GB10

> DO NOT DO THIS.

## What not to do

Do not switch `model-code.sh` to the Intel AutoRound quantization of
Qwen3-Coder-Next (`Intel/Qwen3-Coder-Next-int4-AutoRound`) on the NVIDIA GB10
128GB target.

## Why it is wrong

Commit `2bcfd83` ("Upgrade Coding model to Intel AutoRound") tried the Intel
AutoRound int4 build. Commit `4d8f83e` ("switch back to cyankiwi coding model,
seems to be better for memory with no decline in quality or perf") reverted
it. On this hardware the cyankiwi AWQ-4bit build (`cyankiwi/Qwen3-Coder-Next-AWQ-4bit`)
uses memory better with no quality or performance loss versus the Intel
AutoRound build.

## What to do instead

Keep `model-code.sh` on the cyankiwi AWQ-4bit build:

```bash
ACCT="${ACCT:-cyankiwi}"
MODEL="${MODEL:-Qwen3-Coder-Next-AWQ-4bit}"
```

The commented-out Intel lines in `model-code.sh` are retained as a reminder of
what was tried and rejected — do not uncomment them on GB10 without
re-benchmarking memory and quality.

## Origin

- Revert commit: `4d8f83e` — "chore(model-code) switch back to cyankiwi coding model, seems to be better for memory with no decline in quality or perf"
- Original attempt: `2bcfd83` — "chore(model) Upgrade Coding model to Intel AutoRound"
