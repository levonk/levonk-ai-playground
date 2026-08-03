# Improvement: Reconcile README model table with actual filenames

## Current state

The README model table lists `model-reasoning.sh`, but the actual file is
`model-reason.sh`. The table also lists a different account/model for the
reasoning model (`hesamation` / `Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled`)
than the script uses (`codgician` /
`Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-int4`).

## Proposed change

Update the README model table to match the scripts exactly: filename, `ACCT`,
`MODEL`, `HOST_PORT`, and special config. Re-derive the table from the scripts
rather than maintaining it by hand, or add a comment marking the table as
authoritative so future edits keep it in sync.

## Origin

Discovered during agent-file-upsert on 2026-08-01 while reading the model
scripts against the README.
