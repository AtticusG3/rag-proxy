"""Upstream response parsing for transcript capture."""

import json

from rag_proxy.response_parse import parse_chat_completion


def test_parse_openai_json_extracts_assistant_message_and_usage():
    """Fine-tuning records need the completed assistant message, not raw JSON."""
    body = json.dumps(
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Restart the service."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
    ).encode()

    parsed = parse_chat_completion("v1/chat/completions", body, stream=False)

    assert parsed.raw_ok
    assert parsed.assistant_message == {"role": "assistant", "content": "Restart the service."}
    assert parsed.assistant_text == "Restart the service."
    assert parsed.finish_reason == "stop"
    assert parsed.usage == {"prompt_tokens": 10, "completion_tokens": 4}


def test_parse_openai_sse_aggregates_content_and_tool_arguments():
    """Streaming deltas must reassemble to the same shape as non-stream completions."""
    events = [
        {"choices": [{"delta": {"role": "assistant", "content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo "}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"q"'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": ':"x"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    chunks = [f"data: {json.dumps(event)}\n\n".encode() for event in events]
    chunks.append(b"data: [DONE]\n\n")

    parsed = parse_chat_completion("v1/chat/completions", b"".join(chunks), stream=True)

    assert parsed.raw_ok
    assert parsed.assistant_text == "Hello "
    assert parsed.finish_reason == "tool_calls"
    assert parsed.assistant_message["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": '{"q":"x"}',
    }


def test_parse_ollama_ndjson_aggregates_streamed_message_content():
    """Ollama /api/chat streaming returns NDJSON that still represents one turn."""
    body = b"\n".join(
        [
            b'{"message":{"role":"assistant","content":"part "},"done":false}',
            b'{"message":{"role":"assistant","content":"two"},"done":true,"done_reason":"stop"}',
        ]
    )

    parsed = parse_chat_completion("api/chat", body, stream=True)

    assert parsed.raw_ok
    assert parsed.assistant_message == {"role": "assistant", "content": "part two"}
    assert parsed.finish_reason == "stop"


def test_parse_failure_is_reported_without_raising():
    """Capture must remain fail-open when upstream returns unexpected content."""
    parsed = parse_chat_completion("v1/chat/completions", b"not json", stream=False)

    assert not parsed.raw_ok
    assert parsed.assistant_message is None
    assert parsed.parse_error
