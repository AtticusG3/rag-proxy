"""Transcript message sanitization should remove proxy-only context."""

from rag_proxy.legacy_rag import inject_context
from rag_proxy.message_sanitize import (
    is_exportable_turn,
    sanitize_client_messages,
    strip_rolling_memory,
)


def test_sanitize_client_messages_removes_rag_prefix_but_keeps_system_prompt():
    """Fine-tuning data must not learn retrieved context as user-authored input."""
    messages = [
        {"role": "system", "content": "base instructions"},
        {"role": "user", "content": "how do I restart it?"},
    ]

    out = sanitize_client_messages(inject_context(messages, ["private retrieved chunk"]))

    assert out[0]["role"] == "system"
    assert out[0]["content"] == "base instructions"
    assert "private retrieved chunk" not in out[0]["content"]
    assert out[1]["content"] == "how do I restart it?"


def test_sanitize_client_messages_drops_context_only_system_message():
    """A synthetic RAG-only system message should disappear from transcript input."""
    messages = inject_context([{"role": "user", "content": "hi"}], ["chunk A"])

    out = sanitize_client_messages(messages)

    assert out == [{"role": "user", "content": "hi"}]


def test_strip_rolling_memory_removes_operational_memory_prefix():
    """Rolling memory is runtime context and should not become training text."""
    messages = [
        {
            "role": "system",
            "content": "Operational memory (session):\nold query\n\nbase instructions",
        },
        {"role": "user", "content": "latest"},
    ]

    out = strip_rolling_memory(messages)

    assert out[0]["content"] == "base instructions"


def test_is_exportable_turn_skips_ui_meta_prompt():
    """UI follow-up generation prompts should not become supervised examples."""
    assert not is_exportable_turn(
        "### Task:\nSuggest 3-5 relevant follow-up questions.",
        "What else would you like to know?",
    )
    assert is_exportable_turn("How do I deploy rag-proxy?", "Use systemd and verify logs.")
