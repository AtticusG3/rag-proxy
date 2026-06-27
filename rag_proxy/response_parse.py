"""Parse upstream chat responses into transcript-friendly assistant messages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedCompletion:
    assistant_message: dict[str, Any] | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    raw_ok: bool = False
    parse_error: str | None = None

    @property
    def assistant_text(self) -> str | None:
        if not self.assistant_message:
            return None
        content = self.assistant_message.get("content")
        return content if isinstance(content, str) else None


def parse_chat_completion(path: str, body: bytes, *, stream: bool) -> ParsedCompletion:
    """Parse OpenAI-compatible or Ollama chat responses."""
    route = path.rstrip("/")
    try:
        if route == "v1/chat/completions":
            if stream:
                return _parse_openai_sse(body)
            return _parse_openai_json(body)
        if route == "api/chat":
            if stream:
                return _parse_ollama_ndjson(body)
            return _parse_ollama_json(body)
        return ParsedCompletion(parse_error=f"unsupported chat path: {route}")
    except Exception as e:
        return ParsedCompletion(parse_error=str(e))


def _decode_json_line(raw: bytes) -> dict[str, Any]:
    return json.loads(raw.decode("utf-8"))


def _parse_openai_json(body: bytes) -> ParsedCompletion:
    payload = _decode_json_line(body)
    choices = payload.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message")
    if not isinstance(message, dict):
        return ParsedCompletion(parse_error="missing choices[0].message")
    return ParsedCompletion(
        assistant_message=message,
        finish_reason=first.get("finish_reason"),
        usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
        raw_ok=True,
    )


def _parse_openai_sse(body: bytes) -> ParsedCompletion:
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    for line in body.splitlines():
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[len(b"data:"):].strip()
        if data == b"[DONE]":
            continue
        event = _decode_json_line(data)
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            content_parts.append(content)
        for tool_delta in delta.get("tool_calls") or []:
            if not isinstance(tool_delta, dict):
                continue
            index = int(tool_delta.get("index", 0))
            current = tool_calls.setdefault(index, {"function": {}})
            if tool_delta.get("id"):
                current["id"] = tool_delta["id"]
            if tool_delta.get("type"):
                current["type"] = tool_delta["type"]
            fn_delta = tool_delta.get("function") or {}
            fn_current = current.setdefault("function", {})
            if fn_delta.get("name"):
                fn_current["name"] = fn_delta["name"]
            if isinstance(fn_delta.get("arguments"), str):
                fn_current["arguments"] = fn_current.get("arguments", "") + fn_delta["arguments"]

    message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return ParsedCompletion(
        assistant_message=message,
        finish_reason=finish_reason,
        usage=usage,
        raw_ok=True,
    )


def _parse_ollama_json(body: bytes) -> ParsedCompletion:
    payload = _decode_json_line(body)
    message = payload.get("message")
    if not isinstance(message, dict):
        return ParsedCompletion(parse_error="missing message")
    return ParsedCompletion(
        assistant_message=message,
        finish_reason="stop" if payload.get("done") else None,
        raw_ok=True,
    )


def _parse_ollama_ndjson(body: bytes) -> ParsedCompletion:
    content_parts: list[str] = []
    final: dict[str, Any] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        event = _decode_json_line(line)
        final = event
        message = event.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            content_parts.append(content)
    return ParsedCompletion(
        assistant_message={"role": "assistant", "content": "".join(content_parts)},
        finish_reason="stop" if final.get("done") else None,
        raw_ok=True,
    )
