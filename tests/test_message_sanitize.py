"""Transcript message sanitization should remove proxy-only context."""

from rag_proxy.legacy_rag import inject_context
from rag_proxy.message_sanitize import is_exportable_turn, sanitize_client_messages
from rag_proxy.rag_context import (
    RAG_CONTEXT_PREFIX,
    ROLLING_MEMORY_PREFIX,
    wrap_injected_block,
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


def test_sanitize_client_messages_strips_rolling_memory_prefix():
    """Rolling memory is runtime context and should not become training text."""
    messages = [
        {
            "role": "system",
            "content": "Operational memory (session):\nold query\n\nbase instructions",
        },
        {"role": "user", "content": "latest"},
    ]

    out = sanitize_client_messages(messages)

    assert out[0]["content"] == "base instructions"


def test_round_trip_multi_chunk_rag_block():
    """Blank lines inside chunks must not leave retrieved text behind."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]
    chunks = ["chunk one", "chunk two\n\nmore of two", "chunk three\n\n\nand more"]

    assert sanitize_client_messages(inject_context(messages, chunks)) == messages


def test_round_trip_rolling_memory_block():
    """A marked rolling-memory block with blank lines is stripped whole."""
    block = wrap_injected_block(ROLLING_MEMORY_PREFIX, "old\n\nquery")
    messages = [
        {"role": "system", "content": block + "\n\nYou are helpful."},
        {"role": "user", "content": "hi"},
    ]

    out = sanitize_client_messages(messages)

    assert out[0]["content"] == "You are helpful."


def test_old_unmarked_rag_block_still_stripped():
    """Transcripts written before markers existed keep the first-blank-line rule."""
    messages = [
        {"role": "system", "content": f"{RAG_CONTEXT_PREFIX}old chunk\n\nYou are helpful."},
        {"role": "user", "content": "hi"},
    ]

    out = sanitize_client_messages(messages)

    assert out[0]["content"] == "You are helpful."


def test_is_exportable_turn_skips_ui_meta_prompt():
    """UI follow-up generation prompts should not become supervised examples."""
    assert not is_exportable_turn(
        "### Task:\nSuggest 3-5 relevant follow-up questions.",
        "What else would you like to know?",
    )
    assert is_exportable_turn("How do I deploy rag-proxy?", "Use systemd and verify logs.")
