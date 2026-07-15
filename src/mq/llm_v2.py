"""V2 conversation integration spike.

This module is deliberately separate from mq's current execution path until the
llm_client V2 compatibility contract is complete.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from llm_client import Client, Conversation, ModelResponse


def model_ref(provider: str, model: str) -> str:
    provider = provider.strip().lower()
    if not provider or not model:
        raise ValueError("mq sessions require non-empty provider and model values")
    if model.startswith(f"{provider}/"):
        return model
    return f"{provider}/{model}"


def conversation_from_session(
    session: dict[str, Any],
    client: Client,
    *,
    protocol: str | None = None,
    local_endpoint: str | None = None,
) -> Conversation:
    messages = session.get("messages")
    provider = session.get("provider")
    model = session.get("model")
    if not isinstance(messages, list):
        raise ValueError("mq session messages must be a list")
    if not isinstance(provider, str) or not isinstance(model, str):
        raise ValueError("mq session is missing provider/model identity")

    ref = model_ref(provider, model)
    conversation = Conversation.from_messages(
        messages,
        model=(f"local/unset/{model.split('/')[-1]}" if provider == "local" else ref),
        protocol=protocol,
        metadata={
            "application": {
                "mq": {
                    key: deepcopy(value)
                    for key, value in session.items()
                    if key != "messages"
                }
            }
        },
    )
    conversation.bind(client, local_endpoint=local_endpoint)
    return conversation


def continue_session(
    session: dict[str, Any],
    client: Client,
    prompt: str,
    *,
    protocol: str | None = None,
    local_endpoint: str | None = None,
    **options: Any,
) -> ModelResponse:
    conversation = conversation_from_session(
        session,
        client,
        protocol=protocol,
        local_endpoint=local_endpoint,
    )
    response = conversation.send(prompt, **options)
    session["messages"] = [message.to_standard() for message in conversation.messages]
    session["conversation_v2"] = conversation.to_dict()
    return response
