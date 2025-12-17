from __future__ import annotations

import argparse
import json
import sys
import re
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from llm_client import get_provider

from .errors import LLMError, MQError, UserError
from .llm import chat
from .store import (
    ensure_home,
    create_session,
    get_model,
    list_models,
    list_sessions,
    load_latest_session,
    load_session,
    remove_model,
    rename_session,
    save_session,
    select_session,
    upsert_model,
)

DETAILED_HELP = """\
mq — Model Query CLI

Quickstart:
  mq add gpt --provider openai gpt-4o-mini
  mq query gpt "Write a haiku about recursive functions"
  mq continue "Make it funnier"
  mq batch gpt -i in.jsonl -o out.jsonl

Configuration:
  - Default home: ~/.mq/ (override with MQ_HOME=/path)
  - Model registry: ~/.mq/config.json
  - Sessions: ~/.mq/sessions/<session>.json
  - Latest session pointer: ~/.mq/last_conversation.json (symlink or pointer file)

Commands:
  mq help [topic...]
    - mq help              (this page)
    - mq help query        (subcommand help)
    - mq help session list (nested help)

  mq add <shortname> --provider <provider> <model> [--sysprompt ... | --sysprompt-file PATH]
    - Saves/overwrites a model alias in config.json (no network request).

  mq test <shortname> --provider <provider> <model> [--sysprompt ... | --sysprompt-file PATH] [--json] [--save] "<query>"
    - Validates the provider/model by making a request.
    - By default it does NOT modify config; pass --save to persist/overwrite the alias on success.

  mq models
    - Lists configured shortnames.

  mq query <shortname> [-s/--sysprompt ...] [--json] [-n/--no-session] [--session <id>] "<query>"
  mq ask <shortname> ...          (alias for `mq query`)
  mq q <shortname> ...            (short alias for `mq query`)
    - Runs a one-off query against a configured model.
    - By default creates a new session and prints `session: <id>` first.
    - Use -n/--no-session for ephemeral asks (no session file, no pointer update).
    - Use --session <id> to create a named session (collision = error).

  mq batch <shortname> -i <in.jsonl|-> -o <out.jsonl|->
    - Reads JSONL rows (must include a string `prompt` field), queries the model, and writes JSONL results.
    - Output rows include all input fields plus: `response`, `mq_input_prompt`, `prompt`, and optional `reasoning`/`sysprompt`.
    - Does not create sessions or update ~/.mq/last_conversation.json.
    - Use --workers N for parallelism and --extract-tags to extract <field>value</field> into `tag:field`.
    - Use --prompt to prefix each request: the input row's prompt is appended as an attachment block.
    - Output order matches input order.
    - On row failure, writes `error` (and `error_info` when available); exits non-zero if any row failed.
    - Merge conflicts are fatal (e.g., an input row already contains `response`/`reasoning`/`error` keys, or `tag:*` when --extract-tags is enabled).

  mq continue [--session <id>] [--json] "<query>"
  mq cont [--session <id>] [--json] "<query>"  (alias)
  mq c [--session <id>] [--json] "<query>"     (short alias)
    - Continues a prior session (default: latest).

  mq dump [--session <id>]
    - Dumps a session JSON (default: latest).

  mq session list
  mq session select <id>
  mq session rename <old> <new>
    - Lists/selects/renames sessions. Session ids must match: [A-Za-z0-9][A-Za-z0-9_-]{0,63}

Output formats:
  - Normal: prints `session: <id>` first, then optional reasoning, then response.
  - JSON (--json): single line object containing at least:
      {"response":"...","prompt":"...","session":"..."}
    Optional keys:
      "reasoning" (if provided by model), "sysprompt" (if set), etc.

Provider API keys (environment variables, via llm_client):
  - openai: OPENAI_API_KEY
  - openrouter: OPENROUTER_API_KEY
  - chutes: CHUTES_API_TOKEN

Request controls:
  - -t/--timeout-seconds N  (default: 600)
  - -r/--retries N          (default: 3)
  - Timeout applies per request attempt; retries control how many additional attempts are made for retryable errors.

stdin:
  - For query/continue/test, pass "-" as the query to read the full prompt from stdin.
  - Use --attach PATH to append file contents into the prompt (repeatable; PATH may be "-").
  - stdin can only be consumed once, so you can't combine query "-" with --attach "-".
  - For batch, you may pass '-' to -i/--infile (stdin) and/or -o/--outfile (stdout).
"""

_TAG_RE = re.compile(r"<([A-Za-z0-9_.:-]+)>(.*?)</\1>", re.DOTALL)


def _print_err(message: str) -> None:
    print(message, file=sys.stderr)

def _print_llm_error(error: LLMError) -> None:
    info = error.error_info or {}
    provider = info.get("provider")
    model = info.get("model")
    err_type = info.get("type")
    status_code = info.get("status_code")

    parts: list[str] = []
    if provider:
        parts.append(f"provider={provider}")
    if model:
        parts.append(f"model={model}")
    if err_type:
        parts.append(f"type={err_type}")
    if status_code is not None:
        parts.append(f"status={status_code}")

    prefix = "LLM error"
    if parts:
        prefix += f" ({', '.join(parts)})"

    _print_err(f"{prefix}: {error}")

    snippet = info.get("raw_response_snippet") or info.get("raw_provider_response_snippet")
    if snippet:
        _print_err(f"raw: {snippet}")

def _emit_result(
    *,
    response: str,
    reasoning: str | None,
    json_mode: bool,
    prompt: str | None = None,
    sysprompt: str | None = None,
    session_id: str | None = None,
) -> None:
    if json_mode:
        payload: dict[str, str] = {"response": response}
        if prompt is not None:
            payload["prompt"] = prompt
        if session_id is not None:
            payload["session"] = session_id
        if sysprompt and sysprompt.strip():
            payload["sysprompt"] = sysprompt
        if reasoning and reasoning.strip():
            payload["reasoning"] = reasoning
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return

    if session_id is not None:
        print(f"session: {session_id}")

    if reasoning and reasoning.strip():
        print("reasoning:")
        print(reasoning)
        print()
        print("response:")
    print(response)

def _read_sysprompt_file(path: str) -> str:
    try:
        if path == "-":
            return sys.stdin.read()
        return Path(path).expanduser().read_text(encoding="utf-8")
    except OSError as e:
        raise UserError(f"Failed to read sysprompt file {path!r}: {e}") from e


def _resolve_sysprompt(*, sysprompt: str | None, sysprompt_file: str | None) -> str | None:
    if sysprompt and sysprompt_file:
        raise MQError("Use only one of --sysprompt or --sysprompt-file")
    if sysprompt_file:
        content = _read_sysprompt_file(sysprompt_file)
        return content.rstrip("\n")
    return sysprompt


def _resolve_query(query: str) -> str:
    if query == "-":
        return sys.stdin.read().rstrip("\n")
    return query


def _read_attach(path: str) -> tuple[str, str]:
    if path == "-":
        return "stdin", sys.stdin.read()
    p = Path(path).expanduser()
    try:
        content = p.read_text(encoding="utf-8")
    except OSError as e:
        raise UserError(f"Failed to read attachment {path!r}: {e}") from e
    return p.name, content


def _format_attachment(name: str, content: str) -> str:
    name = name or "attachment"
    return f"--- BEGIN ATTACHMENT: {name} ---\n{content.rstrip()}\n--- END ATTACHMENT: {name} ---"


def _apply_attachments_to_prompt(prompt: str, attach_paths: Iterable[str] | None) -> str:
    paths = [p for p in (attach_paths or []) if p is not None]
    if not paths:
        return prompt
    stdin_count = sum(1 for p in paths if p == "-")
    if stdin_count > 1:
        raise UserError("Only one --attach '-' is allowed")
    if prompt == "-" and stdin_count:
        raise UserError("Cannot use '-' for both query and --attach (stdin can only be consumed once)")

    blocks: list[str] = []
    for path in paths:
        name, content = _read_attach(path)
        blocks.append(_format_attachment(name, content))
    return (prompt.rstrip() + "\n\n" + "\n\n".join(blocks)).rstrip()

def _apply_prompt_prefix(prompt: str, prefix: str | None) -> str:
    if prefix is None or not str(prefix).strip():
        return prompt
    attachment = _format_attachment("input.prompt", prompt)
    return (str(prefix).rstrip() + "\n\n" + attachment).rstrip()


def _extract_tags(text: str) -> dict[str, str | list[str]]:
    matches = list(_TAG_RE.finditer(text or ""))
    if not matches:
        return {}
    extracted: dict[str, list[str]] = {}
    for m in matches:
        name = m.group(1)
        value = (m.group(2) or "").strip()
        extracted.setdefault(name, []).append(value)
    out: dict[str, str | list[str]] = {}
    for name, values in extracted.items():
        key = f"tag:{name}"
        out[key] = values[0] if len(values) == 1 else values
    return out


def _iter_jsonl_objects(fp) -> Iterable[tuple[int, dict]]:
    for line_no, line in enumerate(fp, 1):
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise UserError(f"Invalid JSON on line {line_no}: {e}") from e
        if not isinstance(obj, dict):
            raise UserError(f"JSONL line {line_no} must be an object")
        yield line_no, obj


def _open_text(path: str, mode: str):
    if path == "-":
        return sys.stdin if "r" in mode else sys.stdout
    return Path(path).expanduser().open(mode, encoding="utf-8")


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mq")
    sub = parser.add_subparsers(dest="command", required=True)

    help_cmd = sub.add_parser("help", help="Show detailed help")
    help_cmd.add_argument("topic", nargs=argparse.REMAINDER, help="Optional command path to show help for")

    add = sub.add_parser("add", help="Add/update a model shortname")
    add.add_argument("shortname")
    add.add_argument("--provider", required=True, help="Provider name (llm_client)")
    add.add_argument("model", help="Full model identifier")
    add.add_argument("--sysprompt", help="Saved system prompt for this model")
    add.add_argument("--sysprompt-file", help="Read saved system prompt from file ('-' for stdin)")

    models = sub.add_parser("models", help="List configured models")

    query = sub.add_parser("query", aliases=["ask", "q"], help="Query a configured model")
    query.add_argument("shortname")
    query.add_argument("--sysprompt", "-s", help="Override system prompt for this run")
    query.add_argument("--json", action="store_true", help="Emit a single-line JSON object")
    query.add_argument("-n", "--no-session", action="store_true", help="Do not create or update a session")
    query.add_argument("--session", help="Create a new named session id (collision = error)")
    query.add_argument("--attach", action="append", help="Append file content to the prompt ('-' for stdin)", default=[])
    query.add_argument("-t", "--timeout-seconds", type=_positive_int, help="Request timeout in seconds (default: 600)")
    query.add_argument("-r", "--retries", type=_non_negative_int, help="Max retries for retryable errors (default: 3)")
    query.add_argument("query")

    cont = sub.add_parser("continue", aliases=["cont", "c"], help="Continue the most recent conversation")
    cont.add_argument("--session", help="Continue a specific session id (default: latest)")
    cont.add_argument("--json", action="store_true", help="Emit a single-line JSON object")
    cont.add_argument("--attach", action="append", help="Append file content to the prompt ('-' for stdin)", default=[])
    cont.add_argument("-t", "--timeout-seconds", type=_positive_int, help="Request timeout in seconds (default: 600)")
    cont.add_argument("-r", "--retries", type=_non_negative_int, help="Max retries for retryable errors (default: 3)")
    cont.add_argument("query")

    dump = sub.add_parser("dump", help="Dump the latest session context as JSON")
    dump.add_argument("--session", help="Dump a specific session id (default: latest)")

    rm = sub.add_parser("rm", help="Remove a configured model shortname")
    rm.add_argument("shortname")

    test = sub.add_parser("test", help="Test a provider/model configuration (optionally save with --save)")
    test.add_argument("shortname")
    test.add_argument("--provider", required=True, help="Provider name (llm_client)")
    test.add_argument("model", help="Full model identifier")
    test.add_argument("--sysprompt", help="Saved system prompt for this model")
    test.add_argument("--sysprompt-file", help="Read saved system prompt from file ('-' for stdin)")
    test.add_argument("--json", action="store_true", help="Emit a single-line JSON object")
    test.add_argument("--save", action="store_true", help="Save/overwrite this shortname on success")
    test.add_argument("--attach", action="append", help="Append file content to the prompt ('-' for stdin)", default=[])
    test.add_argument("-t", "--timeout-seconds", type=_positive_int, help="Request timeout in seconds (default: 600)")
    test.add_argument("-r", "--retries", type=_non_negative_int, help="Max retries for retryable errors (default: 3)")
    test.add_argument("query")

    batch = sub.add_parser("batch", help="Process a JSONL file with prompts and write JSONL responses")
    batch.add_argument("shortname")
    batch.add_argument("--infile", "-i", required=True, help="Input JSONL path ('-' for stdin)")
    batch.add_argument("--outfile", "-o", required=True, help="Output JSONL path ('-' for stdout)")
    batch.add_argument("--sysprompt", "-s", help="Override system prompt for this run")
    batch.add_argument("--sysprompt-file", help="Read system prompt from file ('-' for stdin)")
    batch.add_argument("--prompt", help="Prefix prompt (prepends input row prompt as an attachment)", default=None)
    batch.add_argument("--workers", type=_positive_int, default=20, help="Worker threads (default: 20)")
    batch.add_argument("--extract-tags", action="store_true", help="Extract <field>value</field> into `tag:field` keys")
    batch.add_argument("-t", "--timeout-seconds", type=_positive_int, help="Request timeout in seconds (default: 600)")
    batch.add_argument("-r", "--retries", type=_non_negative_int, help="Max retries for retryable errors (default: 3)")

    session = sub.add_parser("session", help="Manage sessions")
    session_sub = session.add_subparsers(dest="session_command", required=True)
    session_sub.add_parser("list", help="List sessions")
    session_select = session_sub.add_parser("select", help="Select a session as latest")
    session_select.add_argument("session_id")
    session_rename = session_sub.add_parser("rename", help="Rename a session id (updates latest pointer if needed)")
    session_rename.add_argument("old_id")
    session_rename.add_argument("new_id")

    return parser


def _cmd_add(args: argparse.Namespace) -> int:
    ensure_home()
    try:
        get_provider(args.provider)
    except Exception as e:
        _print_err(str(e))
        return 2
    sysprompt = _resolve_sysprompt(sysprompt=args.sysprompt, sysprompt_file=args.sysprompt_file)
    upsert_model(args.shortname, args.provider, args.model, sysprompt)
    return 0


def _cmd_models(_: argparse.Namespace) -> int:
    ensure_home()
    items = list_models()
    if not items:
        print("(no models configured)")
        return 0
    for shortname, entry in items:
        provider = entry["provider"]
        model = entry["model"]
        print(f"{shortname}\t{provider}\t{model}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    ensure_home()
    model_cfg = get_model(args.shortname)
    provider = model_cfg["provider"]
    model = model_cfg["model"]
    sysprompt = args.sysprompt if args.sysprompt is not None else model_cfg.get("sysprompt")

    messages: list[dict] = []
    if sysprompt:
        messages.append({"role": "system", "content": sysprompt})
    raw_query = _resolve_query(args.query)
    query = _apply_attachments_to_prompt(raw_query, args.attach)
    messages.append({"role": "user", "content": query})

    result = chat(provider, model, messages, timeout_seconds=args.timeout_seconds, max_retries=args.retries)

    if args.no_session:
        if args.session:
            raise MQError("Cannot use --session with -n/--no-session")
        _emit_result(
            response=result.content,
            reasoning=result.reasoning,
            json_mode=args.json,
            prompt=query,
            sysprompt=sysprompt,
            session_id="(none)" if not args.json else None,
        )
        return 0

    messages.append({"role": "assistant", "content": result.content})
    session_id = create_session(
        model_shortname=args.shortname,
        provider=provider,
        model=model,
        sysprompt=sysprompt,
        messages=messages,
        session_id=args.session,
    )
    _emit_result(
        response=result.content,
        reasoning=result.reasoning,
        json_mode=args.json,
        prompt=query,
        sysprompt=sysprompt,
        session_id=session_id,
    )
    return 0


def _cmd_continue(args: argparse.Namespace) -> int:
    ensure_home()
    session = load_session(args.session) if args.session else load_latest_session()
    provider = session.get("provider")
    model = session.get("model")
    messages = session.get("messages")
    if not isinstance(provider, str) or not isinstance(model, str) or not isinstance(messages, list):
        _print_err("Invalid last conversation format")
        return 2

    if args.json:
        _print_err("warning: --json output does not include full conversation context (use `mq dump` for history)")

    messages = list(messages)
    raw_query = _resolve_query(args.query)
    query = _apply_attachments_to_prompt(raw_query, args.attach)
    messages.append({"role": "user", "content": query})

    result = chat(provider, model, messages, timeout_seconds=args.timeout_seconds, max_retries=args.retries)
    messages.append({"role": "assistant", "content": result.content})
    session["messages"] = messages
    save_session(session)
    _emit_result(
        response=result.content,
        reasoning=result.reasoning,
        json_mode=args.json,
        prompt=query,
        sysprompt=session.get("sysprompt") if isinstance(session, dict) else None,
        session_id=session.get("id") if isinstance(session.get("id"), str) else None,
    )
    return 0


def _cmd_dump(args: argparse.Namespace) -> int:
    ensure_home()
    session = load_session(args.session) if args.session else load_latest_session()
    print(json.dumps(session, indent=2, ensure_ascii=False))
    return 0


def _cmd_rm(args: argparse.Namespace) -> int:
    ensure_home()
    remove_model(args.shortname)
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    ensure_home()
    try:
        get_provider(args.provider)
    except Exception as e:
        _print_err(str(e))
        return 2

    messages: list[dict] = []
    sysprompt = _resolve_sysprompt(sysprompt=args.sysprompt, sysprompt_file=args.sysprompt_file)
    if sysprompt:
        messages.append({"role": "system", "content": sysprompt})
    raw_query = _resolve_query(args.query)
    query = _apply_attachments_to_prompt(raw_query, args.attach)
    messages.append({"role": "user", "content": query})

    result = chat(args.provider, args.model, messages, timeout_seconds=args.timeout_seconds, max_retries=args.retries)
    _emit_result(
        response=result.content,
        reasoning=result.reasoning,
        json_mode=args.json,
        prompt=query,
        sysprompt=sysprompt,
    )

    if args.save:
        upsert_model(args.shortname, args.provider, args.model, sysprompt)
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    ensure_home()
    model_cfg = get_model(args.shortname)
    provider = model_cfg["provider"]
    model = model_cfg["model"]
    sysprompt = _resolve_sysprompt(sysprompt=args.sysprompt, sysprompt_file=args.sysprompt_file)
    if sysprompt is None:
        sysprompt = model_cfg.get("sysprompt")

    any_errors = False

    reserved_output_keys = {"response", "reasoning", "sysprompt", "error", "error_info", "mq_input_prompt"}
    if args.extract_tags:
        # Tag extraction writes keys like `tag:field`, so reserve that namespace.
        reserved_prefixes = ("tag:",)
    else:
        reserved_prefixes = ()

    def process_row(line_no: int, row: dict) -> dict:
        nonlocal any_errors

        prompt_val = row.get("prompt")
        if not isinstance(prompt_val, str):
            any_errors = True
            out = dict(row)
            out["error"] = "Row is missing required string field: prompt"
            return out

        # Hard-fail merge conflicts with reserved output keys.
        for k in reserved_output_keys:
            if k in row:
                raise UserError(f"Batch merge conflict on line {line_no}: input contains reserved key {k!r}")
        for prefix in reserved_prefixes:
            for k in row.keys():
                if isinstance(k, str) and k.startswith(prefix):
                    raise UserError(
                        f"Batch merge conflict on line {line_no}: input contains reserved key prefix {prefix!r} ({k!r})"
                    )

        prompt_final = _apply_prompt_prefix(prompt_val, args.prompt)
        messages: list[dict] = []
        if sysprompt:
            messages.append({"role": "system", "content": sysprompt})
        messages.append({"role": "user", "content": prompt_final})

        out = dict(row)
        out["mq_input_prompt"] = prompt_val
        out["prompt"] = prompt_final
        if sysprompt and sysprompt.strip():
            out["sysprompt"] = sysprompt

        try:
            result = chat(provider, model, messages, timeout_seconds=args.timeout_seconds, max_retries=args.retries)
            out["response"] = result.content
            if result.reasoning and str(result.reasoning).strip():
                out["reasoning"] = result.reasoning
            if args.extract_tags:
                extracted = _extract_tags(result.content)
                for k in extracted.keys():
                    if k in out:
                        raise UserError(f"Batch merge conflict on line {line_no}: extracted key {k!r} already exists")
                out.update(extracted)
        except UserError:
            raise
        except MQError as e:
            any_errors = True
            if isinstance(e, LLMError):
                out["error"] = str(e)
                if e.error_info:
                    out["error_info"] = e.error_info
            else:
                out["error"] = str(e)
        except Exception as e:  # pragma: no cover
            any_errors = True
            out["error"] = f"{type(e).__name__}: {e}"
        return out

    in_fp = _open_text(args.infile, "r")
    close_in = args.infile != "-"
    try:
        rows: list[tuple[int, dict]] = list(_iter_jsonl_objects(in_fp))
    finally:
        if close_in:
            in_fp.close()

    # Detect merge conflicts immediately (before any requests).
    for line_no, row in rows:
        for k in reserved_output_keys:
            if k in row:
                raise UserError(f"Batch merge conflict on line {line_no}: input contains reserved key {k!r}")
        for prefix in reserved_prefixes:
            for k in row.keys():
                if isinstance(k, str) and k.startswith(prefix):
                    raise UserError(
                        f"Batch merge conflict on line {line_no}: input contains reserved key prefix {prefix!r} ({k!r})"
                    )

    # For file outputs, write atomically (avoid partial files on fatal errors).
    temp_out_path: Path | None = None
    if args.outfile != "-":
        out_path = Path(args.outfile).expanduser()
        temp_name = f".{out_path.name}.tmp.{os.getpid()}"
        fd, tmp_path_str = tempfile.mkstemp(prefix=temp_name, dir=str(out_path.parent))
        os.close(fd)
        temp_out_path = Path(tmp_path_str)
        out_fp = temp_out_path.open("w", encoding="utf-8")
        close_out = True
    else:
        out_fp = sys.stdout
        close_out = False

    wrote_all = False
    try:
        results: dict[int, str] = {}
        next_idx = 0

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {}
            for idx, (line_no, row) in enumerate(rows):
                fut = ex.submit(process_row, line_no, row)
                futures[fut] = idx

            try:
                for fut in as_completed(futures):
                    idx = futures[fut]
                    row_out = fut.result()
                    line = json.dumps(row_out, ensure_ascii=False, separators=(",", ":"))
                    results[idx] = line
                    while next_idx in results:
                        out_fp.write(results.pop(next_idx) + "\n")
                        next_idx += 1
            except UserError:
                # Merge conflicts are fatal: cancel pending work and abort.
                for f in futures.keys():
                    f.cancel()
                raise
        wrote_all = True
    finally:
        if close_out:
            out_fp.close()

        if temp_out_path is not None:
            if wrote_all:
                final_out_path = Path(args.outfile).expanduser()
                if final_out_path.exists():
                    final_out_path.unlink()
                temp_out_path.rename(final_out_path)
            else:
                try:
                    temp_out_path.unlink()
                except OSError:
                    pass

    return 1 if any_errors else 0


def _first_user_prompt(messages) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def _cmd_session_list(_: argparse.Namespace) -> int:
    ensure_home()
    sessions = list_sessions()
    if not sessions:
        print("(no sessions)")
        return 0

    def _one_line_prompt(text: str) -> str:
        return (text or "").replace("\n", " ").strip()

    def _format_prompt_preview(prompt: str, max_len: int = 160) -> str:
        prompt = _one_line_prompt(prompt)
        if len(prompt) <= max_len:
            return prompt
        # Show both head and tail for long prompts.
        # Example: "Hello ... do you understand?"
        ellipsis = " ... "
        head_len = (max_len - len(ellipsis)) // 2
        tail_len = max_len - len(ellipsis) - head_len
        head = prompt[:head_len].rstrip()
        tail = prompt[-tail_len:].lstrip()
        return f"{head}{ellipsis}{tail}"

    for s in sessions:
        sid = s.get("id", "")
        updated = s.get("updated_at") or s.get("created_at") or ""
        first = _first_user_prompt(s.get("messages"))
        preview = _format_prompt_preview(first)
        print(f"{sid}\t{updated}")
        print(preview)
    return 0


def _cmd_session_select(args: argparse.Namespace) -> int:
    ensure_home()
    select_session(args.session_id)
    return 0


def _cmd_session_rename(args: argparse.Namespace) -> int:
    ensure_home()
    rename_session(args.old_id, args.new_id)
    return 0


def _cmd_help(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    topic = list(getattr(args, "topic", []) or [])
    if not topic:
        print(DETAILED_HELP, end="")
        return 0
    try:
        # Re-parse with the requested subcommand and --help to leverage argparse output.
        parser.parse_args([*topic, "--help"])
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        match args.command:
            case "help":
                return _cmd_help(args, parser)
            case "add":
                return _cmd_add(args)
            case "models":
                return _cmd_models(args)
            case "query" | "ask" | "q":
                return _cmd_query(args)
            case "continue" | "cont" | "c":
                return _cmd_continue(args)
            case "dump":
                return _cmd_dump(args)
            case "rm":
                return _cmd_rm(args)
            case "test":
                return _cmd_test(args)
            case "batch":
                return _cmd_batch(args)
            case "session":
                match args.session_command:
                    case "list":
                        return _cmd_session_list(args)
                    case "select":
                        return _cmd_session_select(args)
                    case "rename":
                        return _cmd_session_rename(args)
                    case _:
                        _print_err(f"Unknown session command: {args.session_command}")
                        return 2
            case _:
                _print_err(f"Unknown command: {args.command}")
                return 2
    except MQError as e:
        if isinstance(e, LLMError):
            _print_llm_error(e)
        else:
            _print_err(str(e))
        return 2
