import json
from unittest.mock import patch

import httpx

from llm_client import Client, Conversation
from mq.llm import _chat_result, _local_endpoint, chat, model_ref


def test_model_ref_preserves_provider_qualified_models():
    assert model_ref("openrouter", "openai/gpt-5.6-sol") == (
        "openrouter/openai/gpt-5.6-sol"
    )
    assert model_ref("openai", "openai/gpt-5") == "openai/gpt-5"


def test_existing_messages_continue_and_round_trip_canonical_conversation():
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

    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second"},
    ]
    with Client(transport=httpx.MockTransport(handler)) as client:
        conversation = client.model("openrouter/openai/gpt-5.6-sol").conversation(
            messages=messages
        )
        response = conversation.send_pending()
        result = _chat_result(
            response, conversation, "openrouter", "openai/gpt-5.6-sol"
        )

    assert result.content == "second answer"
    assert seen[0]["messages"][-1] == {"role": "user", "content": "second"}
    restored = Conversation.from_dict(result.conversation)
    assert restored.last_operation.status == "succeeded"
    assert restored.to_dict() == result.conversation


def test_local_conversation_does_not_serialize_endpoint():
    conversation = Conversation.from_messages([], model="local/unset/qwen3-4b")
    serialized = conversation.to_dict()
    assert serialized["default_route"]["model"] == "local/unset/qwen3-4b"
    client = Client()
    try:
        try:
            client.load_conversation(serialized)
        except ValueError as error:
            assert "no bound local endpoint" in str(error)
        else:
            raise AssertionError("expected missing endpoint error")
    finally:
        client.close()


def test_local_runtime_endpoint_is_reconstructed_from_mq_model():
    assert _local_endpoint("127.0.0.1:8000/qwen3-4b") == ("http://127.0.0.1:8000/v1")


def test_non_v2_provider_retains_legacy_compatibility_path():
    class Response:
        success = True
        standardized_response = {"content": "legacy", "reasoning": "why"}
        raw_provider_response = {}

    with (
        patch("mq.llm.get_provider", return_value=object()),
        patch("mq.llm.retry_request", return_value=Response()) as retry,
    ):
        result = chat("google", "gemini-test", [{"role": "user", "content": "hello"}])
    assert result.content == "legacy"
    assert result.reasoning == "why"
    assert result.conversation is None
    assert retry.call_args.kwargs["max_retries"] == 3
