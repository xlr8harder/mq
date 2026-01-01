# mq

`mq` is a small CLI for querying configured LLM chat models via `llm_client`.

## Quickstart

```bash
mq add gpt --provider openai gpt-4o-mini
mq query gpt "Write a haiku about recursive functions"
```

`mq query` creates a new session and prints a continuation tip. Continue the latest session with:

```bash
mq continue "Make it funnier"
mq cont "Now in the style of Bash"
mq c "Even shorter continue alias"
```

Aliases for starting a new session:

```bash
mq new gpt "Start fresh"
mq new-session gpt "Start fresh"
```

## Install

From GitHub (pipx recommended):

```bash
pipx install "git+https://github.com/xlr8harder/mq.git"
mq help
```

From a git checkout:

```bash
python -m pip install .
mq help
```

## Configuration

- Home directory: `~/.mq/` (override with `MQ_HOME=/path`)
- Model registry: `~/.mq/config.json`
  - Override registry location with `--config /path/to/config.json` (can appear anywhere on the command line)
- Sessions: `~/.mq/sessions/<session>.json`
- Latest session pointer: `~/.mq/last_conversation.json` (symlink/pointer to latest session file)

## Common Usage

Ephemeral one-off (no session file, no pointer update):

```bash
mq query -n gpt "quick question"
```

Prompt input options:

- Positional query string: `mq query gpt "hello"`
- Read prompt text from a file: `mq query gpt --prompt-file prompt.txt`
- Read prompt text from stdin: `mq query gpt -` or `mq query gpt --prompt-file -`

Attachments:

```bash
mq query -n gpt --attach README.md "Summarize this repo"
cat README.md | mq query -n gpt --attach - "Summarize the attached file"
```

System prompts:

```bash
mq query gpt -s "You are terse and technical." "Explain monads"
mq query gpt --sysprompt-file sysprompt.txt "Explain monads"
```

Script-friendly JSON output:

```bash
mq query gpt --json "Return 3 bullet points"
mq continue --json "Now add a short summary"
```

## Request Controls

Timeout/retries:

```bash
mq query gpt -t 600 -r 3 "slow question"
```

Sampling params (forwarded to the provider; unsupported options will error):

```bash
mq query gpt --temperature 0.7 --top-p 0.95 --top-k 40 "hello"
mq continue --temperature 0.2 "be more deterministic"
mq batch gpt -i in.jsonl -o out.jsonl --temperature 0.7 --progress-seconds 0
```

## Saving Model Defaults

You can store per-alias defaults for sampling params:

```bash
mq add gpt --provider openai gpt-4o-mini --temperature 0.7 --top-p 0.95 --top-k 40
```

You can validate a provider/model configuration with `mq test`:

```bash
mq test --provider openai gpt-4o-mini "hello"
mq test --provider openai gpt-4o-mini --save gpt "hello"  # save on success
```

## Batch JSONL → JSONL

Input format: JSONL with a `prompt` field per row. Output is unordered (completion order) and does not create sessions.

```bash
mq batch gpt -i in.jsonl -o out.jsonl --workers 20 --progress-seconds 0
mq batch gpt -i in.jsonl -o out.jsonl --prompt "You are terse." --extract-tags --progress-seconds 0
```

Each output row merges the input row with: `response`, `mq_input_prompt`, `prompt`, and optionally `sysprompt`/`reasoning`.

## Providers / API Keys

Provider names accepted by `--provider` are implemented in `llm_client`. Common ones:

- `openai` (`OPENAI_API_KEY`)
- `openrouter` (`OPENROUTER_API_KEY`)
- `chutes` (`CHUTES_API_TOKEN`)

Run `mq help` for the most up-to-date list.

## Dev

This project uses `uv`:

```bash
uv run mq --help
uv run python -m unittest discover -s tests -p 'test_*.py'
```
