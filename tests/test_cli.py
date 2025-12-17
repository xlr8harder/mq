import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from mq import cli
from mq import store
from mq.errors import LLMError
from mq.errors import UserError
from mq.llm import ChatResult


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

    def test_ask_creates_new_sessions_and_updates_latest(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)

            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["ask", "m", "Q1"])
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
                    rc = cli.main(["ask", "m", "Q2"])
                self.assertEqual(rc, 0)
                self.assertTrue(out.getvalue().startswith("session: "))
                self.assertTrue(out.getvalue().strip().endswith("A2"))

            sessions2 = store.list_sessions()
            self.assertEqual(len(sessions2), 2)
            latest2 = store.load_latest_session()
            self.assertEqual([m["content"] for m in latest2["messages"]], ["Q2", "A2"])

    def test_continue_appends_and_cont_alias(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)

            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["ask", "m", "Q1"])
                self.assertEqual(rc, 0)

            def fake_chat(provider, model_id, messages):
                self.assertEqual(messages[-1]["role"], "user")
                self.assertEqual(messages[-1]["content"], "Q2")
                return ChatResult(content="A2")

            with patch("mq.cli.chat", side_effect=fake_chat):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["cont", "Q2"])
                self.assertEqual(rc, 0)
                self.assertTrue(out.getvalue().startswith("session: "))
                self.assertTrue(out.getvalue().strip().endswith("A2"))

            latest = store.load_latest_session()
            self.assertEqual([m["content"] for m in latest["messages"]], ["Q1", "A1", "Q2", "A2"])
            self.assertEqual(len(store.list_sessions()), 1)

    def test_sysprompt_override_persists_to_continue(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt="SAVED")

            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cli.main(["ask", "m", "-s", "OVERRIDE", "Q1"])
                self.assertEqual(rc, 0)

            session = store.load_latest_session()
            self.assertEqual(session["sysprompt"], "OVERRIDE")
            self.assertEqual(session["messages"][0]["role"], "system")
            self.assertEqual(session["messages"][0]["content"], "OVERRIDE")

            def fake_chat(provider, model_id, messages):
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
                    cli.main(["ask", "m", "--session", "s1", "Q1"])

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
                rc = cli.main(["ask", "m", "hi"])
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
                rc = cli.main(["ask", "m", "--session", "s1", "Q1"])
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
                rc = cli.main(["ask", "m", "--json", "--session", "s1", "Q1"])
            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue().strip())
            self.assertEqual(payload, {"response": "A1", "prompt": "Q1", "session": "s1"})

    def test_json_output_with_reasoning(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            out = io.StringIO()
            with patch("mq.cli.chat", return_value=ChatResult(content="A1", reasoning="trace")), redirect_stdout(out):
                rc = cli.main(["ask", "m", "--json", "--session", "s1", "Q1"])
            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue().strip())
            self.assertEqual(payload, {"response": "A1", "prompt": "Q1", "session": "s1", "reasoning": "trace"})

    def test_json_output_includes_sysprompt_when_set(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            out = io.StringIO()
            with patch("mq.cli.chat", return_value=ChatResult(content="A1")), redirect_stdout(out):
                rc = cli.main(["ask", "m", "--json", "--session", "s1", "-s", "S", "Q1"])
            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue().strip())
            self.assertEqual(payload, {"response": "A1", "prompt": "Q1", "session": "s1", "sysprompt": "S"})

    def test_continue_json_warns_context_not_included(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out0 = io.StringIO()
                with redirect_stdout(out0):
                    cli.main(["ask", "m", "--session", "s1", "Q1"])

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
                    cli.main(["ask", "m", "--session", "s1", "Q1"])
            sid1 = store.load_latest_session()["id"]
            last_path = store.last_conversation_path()
            self.assertTrue(last_path.exists() or last_path.is_symlink())

            with patch("mq.cli.chat", return_value=ChatResult(content="A2")):
                out2 = io.StringIO()
                with redirect_stdout(out2):
                    cli.main(["ask", "m", "--session", "s2", "Q2"])
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

    def test_ask_no_session_does_not_create_or_update_latest(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out1 = io.StringIO()
                with redirect_stdout(out1):
                    cli.main(["ask", "m", "--session", "s1", "Q1"])
            sid1 = store.load_latest_session()["id"]
            session_count = len(store.list_sessions())

            with patch("mq.cli.chat", return_value=ChatResult(content="A2")):
                out2 = io.StringIO()
                with redirect_stdout(out2):
                    rc = cli.main(["ask", "m", "-n", "Q2"])
            self.assertEqual(rc, 0)
            self.assertTrue(out2.getvalue().startswith("session: (none)\n"))
            self.assertTrue(out2.getvalue().strip().endswith("A2"))
            self.assertEqual(len(store.list_sessions()), session_count)
            self.assertEqual(store.load_latest_session()["id"], sid1)
            self.assertEqual(store.load_last_conversation().get("id"), sid1)

    def test_help_command_prints_detailed_help(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = cli.main(["help"])
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("mq — Model Query CLI", text)
        self.assertIn("mq ask", text)
        self.assertIn("mq session list", text)

    def test_help_topic_forwards_to_argparse_help(self):
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

    def test_session_rename_updates_latest_pointer(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"MQ_HOME": td}, clear=False):
            store.upsert_model("m", "openai", "gpt-4o-mini", sysprompt=None)
            with patch("mq.cli.chat", return_value=ChatResult(content="A1")):
                out1 = io.StringIO()
                with redirect_stdout(out1):
                    cli.main(["ask", "m", "--session", "old", "Q1"])

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
                    cli.main(["ask", "m", "--session", "s1", long_q])
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
