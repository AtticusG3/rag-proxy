"""Metrics enablement and RAG outcome counters."""

from rag_proxy.context import ChunkHit, RequestContext, RetrievalDecision
from rag_proxy.observability import (
    metrics_enabled,
    record_rag_outcome,
    render_metrics_text,
)


def _metric_value(text: str, prefix: str) -> float:
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line.rsplit(" ", 1)[-1])
    return 0.0


def test_metrics_enabled_uses_enable_metrics_flag(monkeypatch):
    monkeypatch.setattr("rag_proxy.config.settings.enable_metrics", True)
    assert metrics_enabled()


def test_record_rag_outcome_counts_attempt_without_chunks():
    before = render_metrics_text()
    ctx = RequestContext(retrieval=RetrievalDecision.FULL, hits=[])
    record_rag_outcome(ctx)
    after = render_metrics_text()
    assert (
        _metric_value(after, 'rag_requests_total{outcome="miss"}')
        == _metric_value(before, 'rag_requests_total{outcome="miss"}') + 1.0
    )
    assert _metric_value(after, "rag_chunks_injected_total") == _metric_value(
        before, "rag_chunks_injected_total"
    )


def test_record_rag_outcome_increments_counters():
    before = render_metrics_text()
    ctx = RequestContext(
        retrieval=RetrievalDecision.FULL,
        hits=[
            ChunkHit(id="h1", text="chunk one", score=0.9),
            ChunkHit(id="h2", text="chunk two", score=0.8),
        ],
    )
    record_rag_outcome(ctx)
    after = render_metrics_text()
    assert (
        _metric_value(after, 'rag_requests_total{outcome="hit"}')
        == _metric_value(before, 'rag_requests_total{outcome="hit"}') + 1.0
    )
    assert _metric_value(after, "rag_chunks_injected_total") == _metric_value(
        before, "rag_chunks_injected_total"
    ) + 2.0


def test_record_rag_outcome_from_context_skip():
    before = render_metrics_text()
    ctx = RequestContext(retrieval=RetrievalDecision.SKIP)
    record_rag_outcome(ctx)
    after = render_metrics_text()
    assert (
        _metric_value(after, 'rag_requests_total{outcome="skip"}')
        == _metric_value(before, 'rag_requests_total{outcome="skip"}') + 1.0
    )
