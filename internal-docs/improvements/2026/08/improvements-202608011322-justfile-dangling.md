# Improvement: Add a justfile or remove dangling just references

## Current state

`devbox.json` declares `just` as a package and maps script names
(`bootstrap`, `doctor`, `clean`, `build`, `lint`, `test`, `typecheck`, `dev`,
`debug`, `install`, `release`) to `just <name>-internal` targets. But there is
no `justfile` in this repo. The targets are inherited from rtk bundles, and it
is unclear which actually resolve.

## Proposed change

Pick one:

- **A.** Add a minimal `justfile` with the targets that make sense for a
  shell-script repo (e.g., `lint` → `shellcheck *.sh`, `test` →
  `./test-query.sh` smoke check) and remove the rest from `devbox.json`.
- **B.** Remove the `scripts` block from `devbox.json` entirely and document
  that this repo has no task runner.

## Origin

Discovered during agent-file-upsert on 2026-08-01 while writing the developer
guide — the template assumed `just build`/`just test` work, but no justfile
exists.
