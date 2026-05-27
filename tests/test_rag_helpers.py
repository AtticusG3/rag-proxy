"""Unit tests for RAG message handling (no network)."""

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rag_proxy", _ROOT / "rag_proxy.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["rag_proxy"] = _mod
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

extract_query_text = _mod.extract_query_text
inject_context = _mod.inject_context


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
