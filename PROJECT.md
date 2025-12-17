# mq — Model Query CLI

## Overview

`mq` is a small command-line tool for querying LLM chat models via the `llm_client` library. It supports:

- Asking a model a one-off question (`mq ask …`)
- Continuing the most recent conversation (`mq continue …` / `mq cont …`)
- Dumping the most recent conversation context (`mq dump`)
- Managing a local registry of model shortnames (`mq add …`, `mq models`)
- Removing model shortnames (`mq rm …`)
- Testing a model configuration before saving (`mq test …`)

## Goals

- Fast CLI for “ask/continue” workflows.
- Simple local configuration with model shortnames.
- Store only the most recent conversation, overwriting on every non-continue run.
- Minimal dependencies; use `llm_client` for provider integration.

## Non-goals (for MVP)

- Storing multiple named conversations.
- Token-by-token streaming output.
- Rich TUI, pagination, or editor integrations.
- Provider-specific advanced options beyond model + system prompt.

## Configuration and Data Storage

### Config directory

On first use, `mq` creates a dotfile directory:

- Default: `~/.mq/`
- Override for testing/automation: `MQ_HOME=/path/to/dir`

### Files

- `~/.mq/config.json`
  - Model registry keyed by shortname.
  - Schema (v1):
    - `version`: integer
    - `models`: map of shortname → `{ provider, model, sysprompt? }`
- `~/.mq/sessions/<id>.json`
  - Each `mq ask` creates a new session file with a random id.
  - Schema (v1):
    - `version`: integer
    - `id`: string
    - `created_at`: string (UTC ISO-ish)
    - `updated_at`: string (UTC ISO-ish)
    - `model_shortname`: string
    - `provider`: string
    - `model`: string
    - `sysprompt`: string | null
    - `messages`: list of OpenAI-style chat messages: `{ role, content }`
- `~/.mq/last_conversation.json`
  - Convenience symlink/pointer to the latest session file.

### System prompt rules

- A model can have a saved `sysprompt` in `mq add`.
- `mq ask` can override system prompt per-run with `--sysprompt/-s`.
- The system prompt used for a conversation is persisted into `last_conversation.json`.
- `mq continue` uses the persisted system prompt and full message history from the last conversation.

## CLI Commands

### `mq add`

Add/update a model shortname:

```
mq add <shortname> --provider <providername> <full/model-name> [--sysprompt "..."]
```

Notes:

- If `<shortname>` exists, the entry is overwritten.
- `<providername>` must be supported by `llm_client.get_provider()`.
- `--sysprompt-file <path>` reads the system prompt from a file (`-` for stdin).

### `mq models`

List configured model shortnames and their provider/model identifiers.

### `mq ask`

Query a configured model:

```
mq ask <shortname> [--sysprompt/-s "..."] "<query>"
```

Behavior:

- Creates a new session from scratch (system prompt + user query).
- Saves it under `~/.mq/sessions/<id>.json` and updates `~/.mq/last_conversation.json`.
- Prints the assistant response to stdout.

Options:

- `-n/--no-session`: do not create a session or update `~/.mq/last_conversation.json` (one-off query).

### `mq continue` / `mq cont`

Continue the most recent conversation:

```
mq continue "<query>"
mq cont "<query>"
```

Behavior:

- Loads the `latest` session (or `--session <id>`), appends the new user message, sends the full history.
- Appends the assistant response and writes the updated session back to disk (and updates `latest`).

### `mq dump`

Dump the last conversational context:

```
mq dump
```

Behavior:

- Prints the `latest` session JSON (or `--session <id>`).
- Exits non-zero if no prior session exists.

### `mq session`

Manage sessions:

```
mq session list
mq session select <id>
```

### `mq rm`

Remove a configured model shortname:

```
mq rm <shortname>
```

### `mq test`

Test a provider/model configuration (and save it on success):

```
mq test <shortname> --provider <providername> <full/model-name> [--sysprompt "..."] "<query>"
mq test <shortname> --provider <providername> <full/model-name> [--sysprompt "..."] --save "<query>"
```

## Error Handling

- Missing config/model shortname: print a clear error to stderr; exit non-zero.
- Missing API keys: surface `llm_client`’s error message; exit non-zero.
- Provider errors: print the standardized error message when available; exit non-zero.
- If the provider returns a separate reasoning trace, print it before the response with a `response:` header separating them.
- `--json` emits a single-line object containing at least `response` and `prompt`; include `reasoning` and `sysprompt` keys when present.

## Testing Requirements

- Unit tests cover:
  - Config creation and read/write round-trips
  - Model registry add/list behavior
  - Conversation overwrite vs continue append semantics
  - CLI argument parsing for `ask`, `continue/cont`, `dump`, `add`
- Tests must not touch the real home directory; use `MQ_HOME` with a temp directory.

## Future Enhancements

- `mq rm <shortname>` and `mq show <shortname>`
- Provider options passthrough (temperature, max_tokens, etc.)
- Structured output (`--json`) for `ask`/`dump`
- Multiple conversation slots / named sessions
