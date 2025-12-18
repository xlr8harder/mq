`mq` is a small CLI for querying configured LLM chat models via `llm_client`.

## Overview

- Configure model aliases (`mq add`) and query them from the terminal (`mq query`).
- Persist multi-turn conversations as sessions (`mq continue`, `mq session list/select/rename`).
- Run ephemeral one-offs without creating sessions (`mq query -n`).
- Script-friendly output (`--json`) and shell ergonomics (`-` for stdin, `--attach` for files).
- Thorough CLI documentation: `mq help` and `mq help <command...>`.

## Examples

Configure a model shortname and ask a question:

```bash
mq add gpt --provider openai gpt-4o-mini
mq query gpt "Write a haiku about recursive functions"
```

Continue the most recent session:

```bash
mq continue "Make it funnier"
mq cont "Now in the style of Bash"
mq c "Even shorter continue alias"
```

Create and name a session (collision = error):

```bash
mq query gpt --session work "Start a tracked conversation"
mq continue --session work "Follow up on that"
mq session rename work work-notes
```

List/select sessions:

```bash
mq session list
mq session select <id>
```

Ephemeral one-off (no session file, no pointer update):

```bash
mq query -n gpt "quick question"
```

Shell ergonomics: stdin prompt + attachments:

```bash
echo "prompt from stdin" | mq query -n gpt -
mq query -n gpt --attach README.md "Summarize this repo"
cat README.md | mq query -n gpt --attach - "Summarize the attached file"
```

Script-friendly JSON output:

```bash
mq query gpt --json "Return 3 bullet points"
mq continue --json "Now add a short summary"
```

Batch JSONL → JSONL processing (no sessions):

Input format: JSONL with a `prompt` field per row.

```bash
mq batch gpt -i in.jsonl -o out.jsonl --workers 20
mq batch gpt -i in.jsonl -o out.jsonl --prompt "You are terse." --extract-tags
mq batch gpt -i in.jsonl -o out.jsonl -t 600 -r 5 --workers 50
```

Behavior:

- Each input row is copied into the output row and merged with:
  - `response` (required)
  - `mq_input_prompt` (the original input row prompt)
  - `prompt` (the exact prompt sent to the model, including any `--prompt` prefix formatting)
  - `sysprompt` (if set)
  - `reasoning` (if returned by the provider)
- Output is written incrementally in completion order (unordered) to support large batch runs.
- No sessions are created/updated.
- If a row fails, it still produces an output row with `error` (and `error_info` when available), and the overall exit code is non-zero.
- Request controls: `-t/--timeout-seconds` and `-r/--retries` apply per row; batch defaults are `-t 600` and `-r 5`.
- Progress: prints periodic progress lines to stderr; disable with `--progress-seconds 0`.
  - For file inputs (non-stdin), progress includes a best-effort ETA (estimated rows remaining based on completed rate; uses file byte position as a proxy for how much input has been ingested).
  - `outstanding` is submitted-but-not-finished rows (can exceed `workers` because `mq batch` buffers submissions to keep workers busy).
  - `input_read` tracks how much of the input file has been consumed; it can reach 100% while requests are still in flight.

Tag extraction:

- With `--extract-tags`, any `<field>value</field>` blocks in the model response are extracted into `tag:field` keys.
- If the same tag appears multiple times, the extracted value is a JSON list.
- `--extract-tags` reserves the `tag:` key namespace; if any input row already contains keys starting with `tag:`, `mq batch` fails fast.

Tune request behavior:

- `-t/--timeout-seconds`: per-request timeout (seconds), default `600`
- `-r/--retries`: max retry attempts for transient errors, default `3`

`-t` applies to each request attempt; `-r` controls how many additional attempts are made when `llm_client` considers an error retryable.

```bash
mq query gpt -t 600 -r 3 "slow question"
```

## Install

From a git checkout:

```bash
python -m pip install .
mq help
```

From GitHub (pipx recommended):

```bash
pipx install "git+https://github.com/xlr8harder/mq.git"
mq help
```

## CLI help

Everything should be discoverable from the CLI:

```bash
mq help
mq help query
mq help session list
mq --help
```

Note: `mq ask` is supported as an alias for `mq query` (examples use `query`).

## Dev / run locally

This project uses `uv`. Typical usage is:

```bash
uv run mq --help
```

You can also run it as a module:

```bash
python -m mq --help
```

## API keys

API keys are read from environment variables (via `llm_client`):

- `openai`: `OPENAI_API_KEY`
- `openrouter`: `OPENROUTER_API_KEY`
- `chutes`: `CHUTES_API_TOKEN`

## Supported providers

Provider names accepted by `--provider` (via `llm_client`):

- `openai`
- `openrouter`
- `fireworks`
- `chutes`
- `google`
- `tngtech`
- `xai`
- `moonshot`

Run `mq help` for the most up-to-date usage and environment variable details.

## Configure a model shortname

```bash
mq add gpt --provider openai gpt-4o-mini
mq models
```

Load a system prompt from a file:

```bash
mq add glm --provider openrouter some/model --sysprompt-file sysprompt.txt
```

Test a configuration before saving:

```bash
mq test gpt --provider openai gpt-4o-mini "hello"
mq test gpt --provider openai gpt-4o-mini --save "hello"  # save on success
```

Configuration is stored under `~/.mq/` by default (override with `MQ_HOME=/path`).

Remove a configured shortname:

```bash
mq rm gpt
```

## Query / continue / dump

```bash
mq query gpt "Write a haiku about recursive functions"
mq continue "Make it funnier"
mq c "Now in the style of Bash"
mq dump
```

One-off ask without creating a session:

```bash
mq query -n gpt "quick question"
echo "prompt from stdin" | mq query -n gpt -
mq query -n gpt --attach README.md "Summarize this repo"
cat README.md | mq query -n gpt --attach - "Summarize the attached file"
```

Override request controls:

```bash
mq query gpt -t 600 -r 3 "slow question"
```

If a provider returns a separate reasoning trace, `mq` prints it before the response, with a `response:` header separating them.
Use `--json` to get a single-line JSON object on stdout including the query prompt: `{"response":"...","prompt":"...","reasoning":"..."}` (omit `reasoning`/`sysprompt` if absent).

## Sessions

Each `mq query` creates a new session under `~/.mq/sessions/`.
For convenience, `~/.mq/last_conversation.json` is maintained as a symlink/pointer to the latest session file.

```bash
mq session list
mq session select <id>
mq session rename <old> <new>
mq continue --session <id> "follow up"
```

Override system prompt at query time:

```bash
mq query gpt -s "You are terse and technical." "Explain monads"
```
