# Improvement: Add shellcheck + shfmt linting

## Current state

There is no linting. The scripts use `set -euo pipefail` and a
`# shellcheck source=` directive on the library source line, but shellcheck is
not run. The test-query scripts are large (200+ lines) with no static check.

## Proposed change

Add a `justfile` target (see the justfile improvement) that runs:

```bash
shellcheck -x *.sh
shfmt -d *.sh
```

Wire it into `devbox.json` by adding `shellcheck` and `shfmt` packages. This
catches unset-variable, quoting, and SC2086 word-splitting bugs before a
failed deploy on the GPU box.

## Origin

Discovered during agent-file-upsert on 2026-08-01 — the repo has no quality
gates at all, and the scripts handle env vars and Docker args where quoting
mistakes are easy to make.
