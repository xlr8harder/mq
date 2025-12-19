# Changelog

## Unreleased

- TBD

## 0.2.0 (2025-12-19)

- Add `mq batch` for concurrent JSONL → JSONL processing with progress reporting.
- Add `mq query` as the primary command (keep `mq ask` as an alias) and short aliases `mq q` / `mq c`.
- Improve CLI ergonomics: `--attach`, stdin `-`, `--sysprompt-file`, better errors, and structured `--json` output.
- Add request controls (`-t/--timeout-seconds`, `-r/--retries`) including batch-tuned defaults.

## 0.1.0

- Initial public release of `mq`.
