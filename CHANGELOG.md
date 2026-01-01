# Changelog

## Unreleased

- Add `--prompt-file` for `mq query`/`mq continue`/`mq test`.
- Allow `--config` to appear anywhere in the command line.
- Add `mq new-session` alias for `mq query`.
- Add `mq new` alias for `mq query`.
- Print a `mq continue` tip after `mq query` (non-JSON output).
- Add `--temperature`/`--top-p`/`--top-k` to query/continue/test/batch, with per-model defaults via `mq add`.
- Change `mq test` to not require a shortname; use `--save <shortname>` to persist on success.

## 0.2.1 (2025-12-19)

- Add global `mq --config PATH ...` to override the model registry location.
- Add `mq query --sysprompt-file` and enforce single-stdin consumption across query/attach/sysprompt-file.
- Improve `mq batch` progress reporting semantics and ETA.

## 0.2.0 (2025-12-19)

- Add `mq batch` for concurrent JSONL → JSONL processing with progress reporting.
- Add `mq query` as the primary command (keep `mq ask` as an alias) and short aliases `mq q` / `mq c`.
- Improve CLI ergonomics: `--attach`, stdin `-`, `--sysprompt-file`, better errors, and structured `--json` output.
- Add request controls (`-t/--timeout-seconds`, `-r/--retries`) including batch-tuned defaults.

## 0.1.0

- Initial public release of `mq`.
