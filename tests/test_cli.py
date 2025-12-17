import io
import json
import os
import tempfile
import unittest
from threading import Event
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from mq import cli
from mq import store
from mq.errors import LLMError
from mq.errors import UserError
from mq.llm import ChatResult
from mq import llm as mq_llm


class MQCLITests(unittest.TestCase):
    def test_dump_errors_without_conversation(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc = cli.main(["dump"])
            self.assertEqual(rc, 2)
            self.assertIn("No previous conversation found", err.getvalue())

    def test_add_and_models(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc = cli.main(["add", "gpt", "--provider", "openai", "gpt-4o-mini"])
            self.assertEqual(rc, 0, err.getvalue())

            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc = cli.main(["models"])
            self.assertEqual(rc, 0, err.getvalue())
            self.assertIn("gpt\topenai\tgpt-4o-mini", out.getvalue().strip())

    def test_add_overwrites_existing_entry(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt="S1")
            store.upsert_model("m", "openrouter", "anthropic/claude-3.5-sonnet", sysprompt="S2")
            entry = store.get_model("m")
            self.assertEqual(entry["provider"], "openrouter")
            self.assertEqual(entry["model"], "anthropic/claude-3.5-sonnet")
            self.assertEqual(entry["sysprompt"], "S2")

    def test_add_sysprompt_file(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            prompt_path = Path(td) / "prompt.txt"
            prompt_path.write_text("HELLO\n", encoding="utf-8")
            rc = cli.main(
                [
                    "add",
                    "m",
                    "--provider",
                    "openai",
                    "gpt-4o-mini",
                    "--sysprompt-file",
                    str(prompt_path),
                ]
            )
            self.assertEqual(rc, 0)
            entry = store.get_model("m")
            self.assertEqual(entry["sysprompt"], "HELLO")

    def test_add_sysprompt_and_file_conflict_errors(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            prompt_path = Path(td) / "prompt.txt"
            prompt_path.write_text("HELLO\n", encoding="utf-8")
            err = io.StringIO()
            with redirect_stderr(err):
                rc = cli.main(
                    [
                        "add",
                        "m",
                        "--provider",
                        "openai",
                        "gpt-4o-mini",
                        "--sysprompt",
                        "X",
                        "--sysprompt-file",
                        str(prompt_path),
                    ]
                )
            self.assertEqual(rc, 2)
            self.assertIn("--sysprompt", err.getvalue())

    def test_query_creates_new_sessions_and_updates_latest(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)

            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["query", "m", "Q1"])
                self.assertEqual(rc, 0)
                self.assertTrue(out.getvalue().startswith("session: "))
                self.assertTrue(out.getvalue().strip().endswith("A1"))

            sessions = store.list_sessions()
            self.assertEqual(len(sessions), 1)
            latest = store.load_latest_session()
            self.assertEqual(latest["model_shortname"], "m")
            self.assertEqual([m["role"] for m in latest["messages"]], ["user", "assistant"])
            self.assertEqual([m["content"] for m in latest["messages"]], ["Q1", "A1"])

            with patch("mq.cli.chat", return_value=ChatResult(content="A2")):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["query", "m", "Q2"])
                self.assertEqual(rc, 0)
                self.assertTrue(out.getvalue().startswith("session: "))
                self.assertTrue(out.getvalue().strip().endswith("A2"))

            sessions2 = store.list_sessions()
            self.assertEqual(len(sessions2), 2)
            latest2 = store.load_latest_session()
            self.assertEqual([m["content"] for m in latest2["messages"]], ["Q2", "A2"])

    def test_continue_appends_and_short_aliases(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)

            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["query", "m", "Q1"])
                self.assertEqual(rc, 0)

            def fake_chat(provider, model_id, messages, **_kwargs):
                last = messages[-1]["content"]
                if last == "Q2":
                    return ChatResult(content="A2")
                if last == "Q3":
                    return ChatResult(content="A3")
                raise AssertionError(f"unexpected user message: {last!r}")

            with patch("mq.cli.chat", side_effect=fake_chat):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["cont", "Q2"])
                self.assertEqual(rc, 0)
                self.assertTrue(out.getvalue().startswith("session: "))
                self.assertTrue(out.getvalue().strip().endswith("A2"))

                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["c", "Q3"])
                self.assertEqual(rc, 0)
                self.assertTrue(out.getvalue().startswith("session: "))
                self.assertTrue(out.getvalue().strip().endswith("A3"))

            latest = store.load_latest_session()
            self.assertEqual([m["content"] for m in latest["messages"]], ["Q1", "A1", "Q2", "A2", "Q3", "A3"])
            self.assertEqual(len(store.list_sessions()), 1)

    def test_sysprompt_override_persists_to_continue(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt="SAVED")

            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["query", "m", "-s", "OVERRIDE", "Q1"])
                self.assertEqual(rc, 0)

            session = store.load_latest_session()
            self.assertEqual(session["sysprompt"], "OVERRIDE")
            self.assertEqual(session["messages"][0]["role"], "system")
            self.assertEqual(session["messages"][0]["content"], "OVERRIDE")

            def fake_chat(provider, model_id, messages, **_kwargs):
                self.assertEqual(messages[0]["role"], "system")
                self.assertEqual(messages[0]["content"], "OVERRIDE")
                return ChatResult(content="A2")

            with patch("mq.cli.chat", side_effect=fake_chat):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["continue", "Q2"])
                self.assertEqual(rc, 0)

            session2 = store.load_latest_session()
            self.assertEqual(session2["sysprompt"], "OVERRIDE")
            self.assertEqual([m["role"] for m in session2["messages"]][-2:], ["user", "assistant"])

    def test_dump_outputs_json(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out = io.StringIO()
                with redirect_stdout(out):
                    cli.main(["query", "m", "--session", "s1", "Q1"])

            out = io.StringIO()
            with redirect_stdout(out):
                rc = cli.main(["dump"])
            self.assertEqual(rc, 0)
            data = json.loads(out.getvalue())
            self.assertEqual(data["model_shortname"], "m")
            self.assertIn("id", data)

    def test_rm_removes_shortname(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            rc = cli.main(["rm", "m"])
            self.assertEqual(rc, 0)
            with self.assertRaises(UserError):
                store.get_model("m")

    def test_test_command_runs_query_and_saves_on_success(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            with patch("mq.cli.chat", return_value=ChatResult(content="OK")):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["test", "m", "--provider", "openai", "gpt-4o-mini", "--save", "hello"])
            self.assertEqual(rc, 0)
            self.assertEqual(out.getvalue().strip(), "OK")
            entry = store.get_model("m")
            self.assertEqual(entry["provider"], "openai")
            self.assertEqual(entry["model"], "gpt-4o-mini")

    def test_test_command_does_not_save_by_default(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            with patch("mq.cli.chat", return_value=ChatResult(content="OK")):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["test", "m", "--provider", "openai", "gpt-4o-mini", "hello"])
            self.assertEqual(rc, 0)
            self.assertEqual(out.getvalue().strip(), "OK")
            with self.assertRaises(UserError):
                store.get_model("m")

    def test_llm_errors_print_diagnostics(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "chutes", "some/model", sysprompt=None)

            def boom(*_args, **_kwargs):
                raise LLMError(
                    "",
                    error_info={
                        "provider": "chutes",
                        "model": "some/model",
                        "type": "api_error",
                        "status_code": 401,
                        "raw_response_snippet": "{\"error\":{\"message\":\"bad key\"}}",
                    },
                )

            err = io.StringIO()
            with patch("mq.cli.chat", side_effect=boom), redirect_stderr(err):
                rc = cli.main(["query", "m", "hi"])
            self.assertEqual(rc, 2)
            self.assertIn("provider=chutes", err.getvalue())
            self.assertIn("status=401", err.getvalue())
            self.assertIn("bad key", err.getvalue())

    def test_reasoning_traces_are_printed_to_stderr(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            out = io.StringIO()
            err = io.StringIO()
            with (
                patch("mq.cli.chat", return_value=ChatResult(content="A1", reasoning="trace")),
                redirect_stdout(out),
                redirect_stderr(err),
            ):
                rc = cli.main(["query", "m", "--session", "s1", "Q1"])
            self.assertEqual(rc, 0)
            text = out.getvalue()
            self.assertIn("session: s1", text)
            self.assertIn("reasoning:", text)
            self.assertIn("trace", text)
            self.assertIn("\n\nresponse:\n", text)
            self.assertTrue(text.strip().endswith("A1"))
            self.assertEqual(err.getvalue(), "")

    def test_json_output_without_reasoning(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            out = io.StringIO()
            with patch("mq.cli.chat", return_value=ChatResult(content="A1")), redirect_stdout(out):
                rc = cli.main(["query", "m", "--json", "--session", "s1", "Q1"])
            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue().strip())
            self.assertEqual(payload, {"response": "A1", "prompt": "Q1", "session": "s1"})

    def test_json_output_with_reasoning(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            out = io.StringIO()
            with patch("mq.cli.chat", return_value=ChatResult(content="A1", reasoning="trace")), redirect_stdout(out):
                rc = cli.main(["query", "m", "--json", "--session", "s1", "Q1"])
            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue().strip())
            self.assertEqual(payload, {"response": "A1", "prompt": "Q1", "session": "s1", "reasoning": "trace"})

    def test_json_output_includes_sysprompt_when_set(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            out = io.StringIO()
            with patch("mq.cli.chat", return_value=ChatResult(content="A1")), redirect_stdout(out):
                rc = cli.main(["query", "m", "--json", "--session", "s1", "-s", "S", "Q1"])
            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue().strip())
            self.assertEqual(payload, {"response": "A1", "prompt": "Q1", "session": "s1", "sysprompt": "S"})

    def test_continue_json_warns_context_not_included(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out0 = io.StringIO()
                with redirect_stdout(out0):
                    cli.main(["query", "m", "--session", "s1", "Q1"])

            out = io.StringIO()
            err = io.StringIO()
            with (
                patch("mq.cli.chat", return_value=ChatResult(content="A2")),
                redirect_stdout(out),
                redirect_stderr(err),
            ):
                rc = cli.main(["continue", "--json", "Q2"])
            self.assertEqual(rc, 0)
            self.assertIn("does not include full conversation context", err.getvalue())
            payload = json.loads(out.getvalue().strip())
            self.assertEqual(payload, {"response": "A2", "prompt": "Q2", "session": "s1"})

    def test_session_list_and_select_and_continue_session(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out1 = io.StringIO()
                with redirect_stdout(out1):
                    cli.main(["query", "m", "--session", "s1", "Q1"])
            sid1 = store.load_latest_session()["id"]
            last_path = store.last_conversation_path()
            self.assertTrue(last_path.exists() or last_path.is_symlink())

            with patch("mq.cli.chat", return_value=ChatResult(content="A2")):
                out2 = io.StringIO()
                with redirect_stdout(out2):
                    cli.main(["query", "m", "--session", "s2", "Q2"])
            sid2 = store.load_latest_session()["id"]
            self.assertNotEqual(sid1, sid2)
            # last_conversation.json points at latest session (either symlink or pointer).
            loaded = store.load_last_conversation()
            self.assertEqual(loaded.get("id"), sid2)

            out = io.StringIO()
            with redirect_stdout(out):
                rc = cli.main(["session", "list"])
            self.assertEqual(rc, 0)
            text = out.getvalue()
            self.assertIn(sid1, text)
            self.assertIn(sid2, text)
            # Two-line format includes prompt preview on the next line.
            self.assertIn("Q1", text)
            self.assertIn("Q2", text)

            rc = cli.main(["session", "select", sid1])
            self.assertEqual(rc, 0)
            self.assertEqual(store.load_latest_session()["id"], sid1)

            with patch("mq.cli.chat", return_value=ChatResult(content="A3")):
                out3 = io.StringIO()
                with redirect_stdout(out3):
                    cli.main(["continue", "--session", sid2, "Q3"])
            self.assertEqual(store.load_latest_session()["id"], sid2)
            s2 = store.load_session(sid2)
            self.assertEqual([m["content"] for m in s2["messages"]][-2:], ["Q3", "A3"])

    def test_query_no_session_does_not_create_or_update_latest(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out1 = io.StringIO()
                with redirect_stdout(out1):
                    cli.main(["query", "m", "--session", "s1", "Q1"])
            sid1 = store.load_latest_session()["id"]
            session_count = len(store.list_sessions())

            with patch("mq.cli.chat", return_value=ChatResult(content="A2")):
                out2 = io.StringIO()
                with redirect_stdout(out2):
                    rc = cli.main(["query", "m", "-n", "Q2"])
            self.assertEqual(rc, 0)
            self.assertTrue(out2.getvalue().startswith("session: (none)\n"))
            self.assertTrue(out2.getvalue().strip().endswith("A2"))
            self.assertEqual(len(store.list_sessions()), session_count)
            self.assertEqual(store.load_latest_session()["id"], sid1)
            self.assertEqual(store.load_last_conversation().get("id"), sid1)

    def test_query_aliases_work(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            with patch("mq.cli.chat", side_effect=[ChatResult(content="A1"), ChatResult(content="A2")]):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["ask", "m", "Q1"])
                self.assertEqual(rc, 0)
                self.assertTrue(out.getvalue().startswith("session: "))

                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["q", "m", "Q2"])
                self.assertEqual(rc, 0)
                self.assertTrue(out.getvalue().startswith("session: "))

    def test_help_command_prints_detailed_help(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = cli.main(["help"])
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("mq — Model Query CLI", text)
        self.assertIn("mq query", text)
        self.assertIn("mq ask", text)
        self.assertIn("mq session list", text)

    def test_help_topic_forwards_to_argparse_help(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = cli.main(["help", "query"])
        self.assertEqual(rc, 0)
        self.assertIn("usage:", out.getvalue())

        out = io.StringIO()
        with redirect_stdout(out):
            rc = cli.main(["help", "ask"])
        self.assertEqual(rc, 0)
        self.assertIn("usage:", out.getvalue())

    def test_help_nested_topic_forwards_to_argparse_help(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = cli.main(["help", "session", "list"])
        self.assertEqual(rc, 0)
        self.assertIn("usage:", out.getvalue())

    def test_batch_merges_rows_and_extracts_tags_and_does_not_create_sessions(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            in_path = Path(td) / "in.jsonl"
            out_path = Path(td) / "out.jsonl"
            in_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": 1, "prompt": "P1"}),
                        json.dumps({"id": 2, "prompt": "P2"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            seen = set()

            def fake_chat(provider, model_id, messages, **_kwargs):
                self.assertEqual(provider, "openai")
                self.assertEqual(model_id, "gpt-4o-mini")
                user = messages[-1]["content"]
                self.assertIn("PREFIX", user)
                self.assertIn("BEGIN ATTACHMENT: input.prompt", user)
                if "P1" in user:
                    seen.add(1)
                    return ChatResult(content="<field>one</field>\nR1")
                if "P2" in user:
                    seen.add(2)
                    return ChatResult(content="R2 <field>two</field>")
                raise AssertionError(f"unexpected prompt: {user!r}")

            with patch("mq.cli.chat", side_effect=fake_chat):
                rc = cli.main(
                    [
                        "batch",
                        "m",
                        "-i",
                        str(in_path),
                        "-o",
                        str(out_path),
                        "--prompt",
                        "PREFIX",
                        "--extract-tags",
                        "--workers",
                        "2",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(seen, {1, 2})
            self.assertEqual(store.list_sessions(), [])

            lines = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines), 2)
            rows = [json.loads(l) for l in lines]
            by_id = {r["id"]: r for r in rows}
            self.assertIn("response", by_id[1])
            self.assertIn("prompt", by_id[1])
            self.assertEqual(by_id[1]["mq_input_prompt"], "P1")
            self.assertEqual(by_id[2]["mq_input_prompt"], "P2")
            self.assertEqual(by_id[1]["tag:field"], "one")
            self.assertEqual(by_id[2]["tag:field"], "two")

    def test_batch_merge_conflict_is_fatal(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            in_path = Path(td) / "in.jsonl"
            out_path = Path(td) / "out.jsonl"
            in_path.write_text(json.dumps({"prompt": "P1", "response": "already"}) + "\n", encoding="utf-8")
            err = io.StringIO()
            with redirect_stderr(err):
                rc = cli.main(["batch", "m", "-i", str(in_path), "-o", str(out_path)])
            self.assertEqual(rc, 2)
            self.assertIn("merge conflict", err.getvalue())
            self.assertFalse(out_path.exists())

    def test_batch_output_is_completion_order_unordered(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            in_path = Path(td) / "in.jsonl"
            out_path = Path(td) / "out.jsonl"
            in_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": 1, "prompt": "P1"}),
                        json.dumps({"id": 2, "prompt": "P2"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            allow_p1 = Event()

            def fake_chat(_provider, _model_id, messages, **_kwargs):
                content = messages[-1]["content"]
                if "P1" in content:
                    allow_p1.wait(timeout=2)
                    return ChatResult(content="R1")
                if "P2" in content:
                    allow_p1.set()
                    return ChatResult(content="R2")
                raise AssertionError(f"unexpected prompt: {content!r}")

            with patch("mq.cli.chat", side_effect=fake_chat):
                rc = cli.main(["batch", "m", "-i", str(in_path), "-o", str(out_path), "--workers", "2"])
            self.assertEqual(rc, 0)
            lines = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["id"], 2)

    def test_batch_timeout_and_retries_defaults_and_overrides(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            in_path = Path(td) / "in.jsonl"
            out_path = Path(td) / "out.jsonl"
            in_path.write_text(json.dumps({"id": 1, "prompt": "P"}) + "\n", encoding="utf-8")

            seen = {}

            def fake_chat(_provider, _model_id, _messages, **kwargs):
                seen["timeout_seconds"] = kwargs.get("timeout_seconds")
                seen["max_retries"] = kwargs.get("max_retries")
                return ChatResult(content="R")

            with patch("mq.cli.chat", side_effect=fake_chat):
                rc = cli.main(["batch", "m", "-i", str(in_path), "-o", str(out_path)])
            self.assertEqual(rc, 0)
            self.assertEqual(seen["timeout_seconds"], 600)
            self.assertEqual(seen["max_retries"], 5)

            in_path.write_text(json.dumps({"id": 1, "prompt": "P"}) + "\n", encoding="utf-8")
            seen = {}
            with patch("mq.cli.chat", side_effect=fake_chat):
                rc = cli.main(["batch", "m", "-i", str(in_path), "-o", str(out_path), "-t", "12", "-r", "0"])
            self.assertEqual(rc, 0)
            self.assertEqual(seen["timeout_seconds"], 12)
            self.assertEqual(seen["max_retries"], 0)


class MQLLMControlsTests(unittest.TestCase):
    def test_llm_chat_defaults_timeout_and_retries(self):
        calls = {}

        def fake_get_provider(_name):
            return object()

        class FakeResp:
            success = True
            standardized_response = {"content": "ok"}
            raw_provider_response = {}
            error_info = None

        def fake_retry_request(provider, messages, model_id, **options):
            calls["timeout"] = options.get("timeout")
            calls["max_retries"] = options.get("max_retries")
            return FakeResp()

        with patch("mq.llm.get_provider", side_effect=fake_get_provider), patch("mq.llm.retry_request", side_effect=fake_retry_request):
            res = mq_llm.chat("openai", "gpt-4o-mini", [{"role": "user", "content": "hi"}])
        self.assertEqual(res.content, "ok")
        self.assertEqual(calls["timeout"], mq_llm.DEFAULT_TIMEOUT_SECONDS)
        self.assertEqual(calls["max_retries"], mq_llm.DEFAULT_MAX_RETRIES)

    def test_llm_chat_overrides_timeout_and_retries(self):
        calls = {}

        def fake_get_provider(_name):
            return object()

        class FakeResp:
            success = True
            standardized_response = {"content": "ok"}
            raw_provider_response = {}
            error_info = None

        def fake_retry_request(provider, messages, model_id, **options):
            calls["timeout"] = options.get("timeout")
            calls["max_retries"] = options.get("max_retries")
            return FakeResp()

        with patch("mq.llm.get_provider", side_effect=fake_get_provider), patch("mq.llm.retry_request", side_effect=fake_retry_request):
            res = mq_llm.chat(
                "openai",
                "gpt-4o-mini",
                [{"role": "user", "content": "hi"}],
                timeout_seconds=12,
                max_retries=0,
            )
        self.assertEqual(res.content, "ok")
        self.assertEqual(calls["timeout"], 12)
        self.assertEqual(calls["max_retries"], 0)

    def test_cli_short_flags_pass_timeout_and_retries(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)

            def fake_chat(provider, model_id, messages, **kwargs):
                self.assertEqual(kwargs.get("timeout_seconds"), 12)
                self.assertEqual(kwargs.get("max_retries"), 0)
                return ChatResult(content="OK")

            out = io.StringIO()
            with patch("mq.cli.chat", side_effect=fake_chat), redirect_stdout(out):
                rc = cli.main(["query", "m", "-n", "-t", "12", "-r", "0", "hi"])
            self.assertEqual(rc, 0)
            self.assertTrue(out.getvalue().strip().endswith("OK"))

    def test_query_dash_reads_from_stdin(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)

            def fake_chat(provider, model_id, messages, **_kwargs):
                self.assertEqual(messages[-1]["role"], "user")
                self.assertEqual(messages[-1]["content"], "FROM STDIN")
                return ChatResult(content="OK")

            out = io.StringIO()
            stdin = io.StringIO("FROM STDIN\n")
            with patch("mq.cli.chat", side_effect=fake_chat), patch("sys.stdin", stdin), redirect_stdout(out):
                rc = cli.main(["query", "m", "-n", "-"])
            self.assertEqual(rc, 0)
            self.assertTrue(out.getvalue().strip().endswith("OK"))

    def test_attach_file_appends_to_prompt(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            attach_path = Path(td) / "a.txt"
            attach_path.write_text("ATTACH\n", encoding="utf-8")

            def fake_chat(provider, model_id, messages, **_kwargs):
                content = messages[-1]["content"]
                self.assertIn("Q", content)
                self.assertIn("BEGIN ATTACHMENT: a.txt", content)
                self.assertIn("ATTACH", content)
                self.assertIn("END ATTACHMENT: a.txt", content)
                return ChatResult(content="OK")

            out = io.StringIO()
            with patch("mq.cli.chat", side_effect=fake_chat), redirect_stdout(out):
                rc = cli.main(["query", "m", "-n", "--attach", str(attach_path), "Q"])
            self.assertEqual(rc, 0)

    def test_attach_dash_reads_from_stdin(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)

            def fake_chat(provider, model_id, messages, **_kwargs):
                content = messages[-1]["content"]
                self.assertIn("Q", content)
                self.assertIn("BEGIN ATTACHMENT: stdin", content)
                self.assertIn("ATTACH", content)
                return ChatResult(content="OK")

            out = io.StringIO()
            stdin = io.StringIO("ATTACH\n")
            with patch("mq.cli.chat", side_effect=fake_chat), patch("sys.stdin", stdin), redirect_stdout(out):
                rc = cli.main(["query", "m", "-n", "--attach", "-", "Q"])
            self.assertEqual(rc, 0)

    def test_session_rename_updates_latest_pointer(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out1 = io.StringIO()
                with redirect_stdout(out1):
                    cli.main(["query", "m", "--session", "old", "Q1"])

            self.assertEqual(store.load_latest_session()["id"], "old")
            rc = cli.main(["session", "rename", "old", "new"])
            self.assertEqual(rc, 0)
            self.assertEqual(store.load_latest_session()["id"], "new")
            self.assertTrue(store.session_path("new").exists())

    def test_session_list_long_prompt_shows_head_and_tail(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            long_q = "Hello, I'd like " + ("x" * 300) + " do you understand?"
            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out1 = io.StringIO()
                with redirect_stdout(out1):
                    cli.main(["query", "m", "--session", "s1", long_q])
            out = io.StringIO()
            with redirect_stdout(out):
                rc = cli.main(["session", "list"])
            self.assertEqual(rc, 0)
            text = out.getvalue()
            self.assertIn("Hello, I'd like", text)
            self.assertIn("do you understand?", text)
            self.assertIn(" ... ", text)


if __name__ == "__main__":
    unittest.main()
