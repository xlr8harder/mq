import json

import httpx

from llm_client import Client, Conversation
from mq.llm_v2 import continue_session, conversation_from_session


def test_existing_mq_session_continues_and_preserves_canonical_conversation():
    session = {
        "version": 1,
        "id": "session-1",
        "provider": "openrouter",
        "model": "openai/gpt-5.6-sol",
        "model_shortname": "sol",
        "sysprompt": "Be concise.",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
        ],
    }
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chat-2",
                "object": "chat.completion",
                "model": "openai/gpt-5.6-sol",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "second answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            },
        )

    with Client(transport=httpx.MockTransport(handler)) as client:
        response = continue_session(session, client, "second")

    assert response.content == "second answer"
    assert seen[0]["messages"][-1] == {"role": "user", "content": "second"}
    assert session["messages"][-1] == {"role": "assistant", "content": "second answer"}
    restored = Conversation.from_dict(session["conversation_v2"])
    assert restored.last_operation.status == "succeeded"
    assert restored.metadata["application"]["mq"]["id"] == "session-1"
    assert restored.to_dict() == session["conversation_v2"]


def test_local_session_requires_runtime_endpoint_without_serializing_it():
    session = {"provider": "local", "model": "qwen3-4b", "messages": []}
    client = Client()
    try:
        try:
            conversation_from_session(session, client)
        except ValueError as error:
            assert "no bound local endpoint" in str(error)
        else:
            raise AssertionError("expected missing endpoint error")
    finally:
        client.close()
