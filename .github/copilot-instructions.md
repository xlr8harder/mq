# Copilot instructions for `mq`

## Project

- Language: Python (>=3.13)
- Env/tooling: `uv` (`pyproject.toml`, `uv.lock`)
- CLI entrypoint: `mq` (`mq/cli.py`)
- Local state: `~/.mq/` by default (override with `MQ_HOME`)

## Issue tracking (CRITICAL)

- Use `bd` (beads) for all work tracking; do not create markdown TODO lists.
- Prefer `bd ready --json` → `bd update <id> --status in_progress` → `bd close <id> --reason "..."`.
- Keep `.beads/issues.jsonl` in sync with code changes.

## Testing

- Tests use stdlib `unittest`.
- Run: `python -m unittest discover -s tests -p 'test*.py'`
- Tests must not touch the real home directory; use `MQ_HOME` pointing to a temp dir.

## Coding conventions

- Keep CLI behavior stable: `mq add`, `mq models`, `mq ask`, `mq continue`/`mq cont`, `mq dump`.
- Store only the most recent conversation (`last_conversation.json`) and overwrite on non-continue runs.
- `--sysprompt/-s` on `mq ask` overrides the configured prompt and must persist for `mq continue`.
