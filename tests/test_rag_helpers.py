"""Unit tests for RAG message handling (no network)."""

from rag_proxy.legacy_rag import extract_query_text, inject_context


def test_extract_query_text_uses_last_user_message():
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "latest question"},
    ]
    assert extract_query_text(messages) == "latest question"


def test_extract_query_text_skips_follow_up_task_prompt():
    messages = [
        {"role": "user", "content": "What are the best tools for tinder?"},
        {"role": "assistant", "content": "Use a knife and dry fluff."},
        {
            "role": "user",
            "content": "### Task:\nSuggest 3-5 relevant follow-up questions the user might ask.",
        },
    ]
    assert extract_query_text(messages) == "What are the best tools for tinder?"


def test_inject_context_inserts_system_when_none():
    messages = [{"role": "user", "content": "hi"}]
    out = inject_context(messages, ["chunk A", "chunk B"])
    assert out[0]["role"] == "system"
    assert "chunk A" in out[0]["content"]
    assert "chunk B" in out[0]["content"]
    assert out[1] == messages[0]


def test_inject_context_prefixes_existing_system():
    messages = [
        {"role": "system", "content": "base instructions"},
        {"role": "user", "content": "hi"},
    ]
    out = inject_context(messages, ["retrieved"])
    assert out[0]["role"] == "system"
    assert "retrieved" in out[0]["content"]
    assert "base instructions" in out[0]["content"]
