from __future__ import annotations

import json
from dataclasses import dataclass

from llm_client import get_provider
from llm_client.retry import retry_request

from .errors import LLMError


DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_MAX_RETRIES = 3


@dataclass(frozen=True)
class ChatResult:
    content: str
    reasoning: str | None = None


def _truncate(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _json_snippet(obj, limit: int = 800) -> str:
    if obj is None:
        return ""
    try:
        text = json.dumps(obj, ensure_ascii=False)
    except Exception:
        text = repr(obj)
    return _truncate(text, limit=limit)


def _extract_reasoning(raw_provider_response) -> str | None:
    if not isinstance(raw_provider_response, dict):
        return None

    # Some providers include a top-level reasoning field.
    for key in ("reasoning", "reasoning_content", "thinking", "thoughts"):
        value = raw_provider_response.get(key)
        if isinstance(value, str) and value.strip():
            return value

    choices = raw_provider_response.get("choices")
    if not (isinstance(choices, list) and choices):
        return None

    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return None

    for key in ("reasoning", "thinking", "thoughts"):
        value = choice0.get(key)
        if isinstance(value, str) and value.strip():
            return value

    msg = choice0.get("message")
    if isinstance(msg, dict):
        for key in ("reasoning", "reasoning_content", "thinking", "thoughts"):
            value = msg.get(key)
            if isinstance(value, str) and value.strip():
                return value

        # OpenAI-style structured content blocks (rare, but seen in some gateways).
        content = msg.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in ("reasoning", "thinking"):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
            if parts:
                return "\n".join(parts)

    return None


def _coerce_content(raw_content) -> str | None:
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in ("text", "output_text"):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
        joined = "".join(parts)
        return joined if joined.strip() else None
    return None


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
    provider = get_provider(provider_name)
    timeout = DEFAULT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    retries = DEFAULT_MAX_RETRIES if max_retries is None else max_retries
    options: dict[str, object] = {}
    if temperature is not None:
        options["temperature"] = float(temperature)
    if top_p is not None:
        options["top_p"] = float(top_p)
    if top_k is not None:
        options["top_k"] = int(top_k)

    response = retry_request(
        provider,
        messages=messages,
        model_id=model_id,
        timeout=timeout,
        max_retries=retries,
        **options,
    )
    if not response.success:
        base_error_info = response.error_info or {}
        status_code = base_error_info.get("status_code")
        raw_response = base_error_info.get("raw_response")
        raw_provider_response = response.raw_provider_response

        message = (base_error_info.get("message") or "").strip()
        if not message:
            message = "LLM request failed"

        if status_code and message.startswith("Error (HTTP unknown):"):
            suffix = message.split(":", 1)[1] if ":" in message else ""
            message = f"Error (HTTP {status_code}):{suffix}"

        error_info = dict(base_error_info)
        error_info["provider"] = provider_name
        error_info["model"] = model_id
        if raw_response:
            error_info["raw_response_snippet"] = _truncate(str(raw_response))
        if raw_provider_response is not None:
            error_info["raw_provider_response_snippet"] = _json_snippet(raw_provider_response)

        raise LLMError(message, error_info=error_info)
    standardized = response.standardized_response or {}
    content = _coerce_content(standardized.get("content"))
    if content is None:
        raise LLMError(
            "LLM response missing content",
            error_info={
                "provider": provider_name,
                "model": model_id,
                "standardized_response_snippet": _json_snippet(standardized),
                "raw_provider_response_snippet": _json_snippet(response.raw_provider_response),
            },
        )
    reasoning = _extract_reasoning(response.raw_provider_response)
    return ChatResult(content=content, reasoning=reasoning)
