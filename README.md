`mq` is a small CLI for querying configured LLM chat models via `llm_client`.

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
```

Override request controls:

```bash
mq ask gpt --timeout-seconds 600 --retries 3 "slow question"
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
