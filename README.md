`mq` is a small CLI for querying configured LLM chat models via `llm_client`.

## Overview

- Configure model aliases (`mq add`) and query them from the terminal (`mq ask`).
- Persist multi-turn conversations as sessions (`mq continue`, `mq session list/select/rename`).
- Run ephemeral one-offs without creating sessions (`mq ask -n`).
- Script-friendly output (`--json`) and shell ergonomics (`-` for stdin, `--attach` for files).
- Thorough CLI documentation: `mq help` and `mq help <command...>`.

## Examples

Configure a model shortname and ask a question:

```bash
mq add gpt --provider openai gpt-4o-mini
mq ask gpt "Write a haiku about recursive functions"
```

Continue the most recent session:

```bash
mq continue "Make it funnier"
mq cont "Now in the style of Bash"
```

Create and name a session (collision = error):

```bash
mq ask gpt --session work "Start a tracked conversation"
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
mq ask -n gpt "quick question"
```

Shell ergonomics: stdin prompt + attachments:

```bash
echo "prompt from stdin" | mq ask -n gpt -
mq ask -n gpt --attach README.md "Summarize this repo"
cat README.md | mq ask -n gpt --attach - "Summarize the attached file"
```

Script-friendly JSON output:

```bash
mq ask gpt --json "Return 3 bullet points"
mq continue --json "Now add a short summary"
```

Tune request behavior:

- `-t/--timeout-seconds`: request timeout (seconds), default `600`
- `-r/--retries`: max retries for retryable errors, default `3`

```bash
mq ask gpt -t 600 -r 3 "slow question"
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
mq help ask
mq help session list
mq --help
```

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

## Ask / continue / dump

```bash
mq ask gpt "Write a haiku about recursive functions"
mq continue "Make it funnier"
mq cont "Now in the style of Bash"
mq dump
```

One-off ask without creating a session:

```bash
mq ask -n gpt "quick question"
echo "prompt from stdin" | mq ask -n gpt -
mq ask -n gpt --attach README.md "Summarize this repo"
cat README.md | mq ask -n gpt --attach - "Summarize the attached file"
```

Override request controls:

```bash
mq ask gpt -t 600 -r 3 "slow question"
```

If a provider returns a separate reasoning trace, `mq` prints it before the response, with a `response:` header separating them.
Use `--json` to get a single-line JSON object on stdout including the query prompt: `{"response":"...","prompt":"...","reasoning":"..."}` (omit `reasoning`/`sysprompt` if absent).

## Sessions

Each `mq ask` creates a new session under `~/.mq/sessions/`.
For convenience, `~/.mq/last_conversation.json` is maintained as a symlink/pointer to the latest session file.

```bash
mq session list
mq session select <id>
mq session rename <old> <new>
mq continue --session <id> "follow up"
```

Override system prompt at query time:

```bash
mq ask gpt -s "You are terse and technical." "Explain monads"
```
