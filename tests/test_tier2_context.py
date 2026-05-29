"""Tests for context budget estimation."""

from rag_proxy.config import settings
from rag_proxy.stages.tier2_context import estimate_message_chars


def test_estimate_message_chars_clamps_unbroken_text(monkeypatch):
    monkeypatch.setattr(settings, "enable_tokenizer_estimate", True)
    log_line = "ERROR: " + "x" * 500
    total = estimate_message_chars([{"role": "user", "content": log_line}])
    assert total >= len(log_line)
