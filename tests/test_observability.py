"""Metrics enablement and RAG outcome counters."""

from rag_proxy.observability import (
    metrics_enabled,
    record_rag_outcome,
    render_metrics_text,
)


def test_metrics_enabled_uses_enable_metrics_flag(monkeypatch):
    monkeypatch.setattr("rag_proxy.config.settings.enable_metrics", True)
    monkeypatch.setattr("rag_proxy.config.settings.metrics_port", 0)
    assert metrics_enabled()


def test_record_rag_outcome_counts_attempt_without_chunks(monkeypatch):
    monkeypatch.setattr("rag_proxy.observability._requests_total", 0)
    monkeypatch.setattr("rag_proxy.observability._chunks_injected_total", 0)
    record_rag_outcome(0)
    text = render_metrics_text()
    assert "rag_requests_total 1" in text
    assert "rag_chunks_injected_total 0" in text


def test_record_rag_outcome_increments_counters(monkeypatch):
    monkeypatch.setattr("rag_proxy.observability._requests_total", 0)
    monkeypatch.setattr("rag_proxy.observability._chunks_injected_total", 0)
    record_rag_outcome(2)
    text = render_metrics_text()
    assert "rag_requests_total 1" in text
    assert "rag_chunks_injected_total 2" in text
