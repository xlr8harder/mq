from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from llm_client import (
    Client,
    CodexOAuthManager,
    Conversation,
    ModelResponse,
    get_provider,
)
from llm_client.retry import retry_request

from .errors import LLMError


DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_MAX_RETRIES = 3
V2_PROVIDERS = frozenset({"codex", "local", "openai", "openrouter"})


@dataclass(frozen=True)
class ChatResult:
    content: str
    reasoning: str | None = None
    conversation: dict[str, Any] | None = None


def model_ref(provider: str, model: str) -> str:
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        raise ValueError("mq requires non-empty provider and model values")
    if model.startswith(f"{provider}/"):
        return model
    return f"{provider}/{model}"


def chat(
    provider_name: str,
    model_id: str,
    messages: list[dict],
    *,
    timeout_seconds: int | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> ChatResult:
    timeout = DEFAULT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    retries = DEFAULT_MAX_RETRIES if max_retries is None else max_retries
    options = _sampling_options(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    if provider_name.strip().lower() not in V2_PROVIDERS:
        return _legacy_chat(
            provider_name,
            model_id,
            messages,
            timeout=timeout,
            max_retries=retries,
            **options,
        )
    with _client(provider_name, timeout=timeout, max_retries=retries) as client:
        conversation = client.model(model_ref(provider_name, model_id)).conversation(
            messages=messages
        )
        response = conversation.send_pending(**options)
        return _chat_result(response, conversation, provider_name, model_id)


def continue_conversation(
    serialized: dict[str, Any],
    prompt: str,
    *,
    provider_name: str,
    model_id: str,
    timeout_seconds: int | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    local_endpoint: str | None = None,
) -> ChatResult:
    timeout = DEFAULT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    retries = DEFAULT_MAX_RETRIES if max_retries is None else max_retries
    options = _sampling_options(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    if provider_name.strip().lower() == "local" and local_endpoint is None:
        local_endpoint = _local_endpoint(model_id)
    with _client(
        provider_name,
        timeout=timeout,
        max_retries=retries,
        local_endpoint=local_endpoint,
    ) as client:
        conversation = client.load_conversation(
            serialized, local_endpoint=local_endpoint
        )
        expected = model_ref(provider_name, model_id)
        bound = conversation.default_route.get("model")
        if bound != expected and not (
            provider_name == "local"
            and isinstance(bound, str)
            and bound == f"local/unset/{model_id.split('/')[-1]}"
        ):
            raise LLMError(
                "Saved conversation route does not match the MQ session model",
                error_info={
                    "provider": provider_name,
                    "model": model_id,
                    "conversation_model": bound,
                },
            )
        response = conversation.send(prompt, **options)
        return _chat_result(response, conversation, provider_name, model_id)


def _sampling_options(**values: float | int | None) -> dict[str, float | int]:
    return {key: value for key, value in values.items() if value is not None}


def _legacy_chat(
    provider_name: str,
    model_id: str,
    messages: list[dict],
    *,
    timeout: int,
    max_retries: int,
    **options: Any,
) -> ChatResult:
    response = retry_request(
        get_provider(provider_name),
        messages=messages,
        model_id=model_id,
        timeout=timeout,
        max_retries=max_retries,
        **options,
    )
    if not response.success:
        info = dict(response.error_info or {})
        info.update({"provider": provider_name, "model": model_id})
        message = (info.get("message") or "LLM request failed").strip()
        status_code = info.get("status_code")
        if status_code and message.startswith("Error (HTTP unknown):"):
            suffix = message.split(":", 1)[1] if ":" in message else ""
            message = f"Error (HTTP {status_code}):{suffix}"
        if response.raw_provider_response is not None:
            info["raw_provider_response_snippet"] = _json_snippet(
                response.raw_provider_response
            )
        raise LLMError(message, error_info=info)
    standardized = response.standardized_response or {}
    content = _coerce_content(standardized.get("content"))
    if content is None:
        raise LLMError(
            "LLM response missing content",
            error_info={
                "provider": provider_name,
                "model": model_id,
                "standardized_response_snippet": _json_snippet(standardized),
            },
        )
    reasoning = standardized.get("reasoning") or _extract_reasoning(
        response.raw_provider_response
    )
    return ChatResult(
        content=content,
        reasoning=reasoning if isinstance(reasoning, str) else None,
    )


def _coerce_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return None
    parts = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
            text = item.get("text") or item.get("content")
            if isinstance(text, str):
                parts.append(text)
    joined = "".join(parts)
    return joined if joined.strip() else None


def _extract_reasoning(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    containers = [raw]
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        containers.append(choices[0])
        message = choices[0].get("message")
        if isinstance(message, dict):
            containers.append(message)
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") in {
                        "reasoning",
                        "thinking",
                    }:
                        text = item.get("text") or item.get("content")
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
                if parts:
                    return "\n".join(parts)
    for container in containers:
        for key in ("reasoning", "reasoning_content", "thinking", "thoughts"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def codex_oauth_manager(*, client_id: str | None = None) -> CodexOAuthManager:
    resolved = client_id or os.getenv("LLM_CLIENT_CODEX_CLIENT_ID")
    if not resolved:
        raise LLMError(
            "Codex requires LLM_CLIENT_CODEX_CLIENT_ID. Set it, then run `mq auth login codex`."
        )
    return CodexOAuthManager.create(client_id=resolved)


def _local_endpoint(model: str) -> str:
    value = model.removeprefix("local/")
    host, separator, _model_id = value.partition("/")
    if not separator or not host or host == "unset":
        raise LLMError(
            "Local sessions require a runtime route such as 127.0.0.1:8000/model. "
            "Update the MQ model alias before continuing this session."
        )
    return host if "://" in host else f"http://{host}/v1"


@contextmanager
def _client(provider: str, **kwargs: Any):
    manager = codex_oauth_manager() if provider.strip().lower() == "codex" else None
    client = Client(auth={"codex": manager} if manager else None, **kwargs)
    try:
        yield client
    finally:
        client.close()
        if manager is not None:
            manager.close()


def _chat_result(
    response: ModelResponse,
    conversation: Conversation,
    provider: str,
    model: str,
) -> ChatResult:
    if not response.ok:
        error = response.error
        info: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "type": error.category if error else "api_error",
            "status_code": error.status_code if error else None,
            "retryable": error.retryable if error else False,
        }
        if response.raw is not None:
            info["raw_provider_response_snippet"] = _json_snippet(response.raw)
        raise LLMError(
            error.message if error and error.message else "LLM request failed",
            error_info=info,
        )
    if not isinstance(response.content, str):
        raise LLMError(
            "LLM response missing content",
            error_info={
                "provider": provider,
                "model": model,
                "raw_provider_response_snippet": _json_snippet(response.raw),
            },
        )
    return ChatResult(
        content=response.content,
        reasoning=response.reasoning,
        conversation=conversation.to_dict(),
    )


def _json_snippet(obj: Any, limit: int = 800) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False)
    except Exception:
        text = repr(obj)
    return text if len(text) <= limit else text[:limit] + "..."
