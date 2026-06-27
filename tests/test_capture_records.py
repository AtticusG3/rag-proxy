"""Transcript capture record contracts."""

from rag_proxy import capture
from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, RequestContext
from rag_proxy.response_parse import ParsedCompletion


def _ctx() -> RequestContext:
    return RequestContext(
        query_text="How do I deploy rag-proxy?",
        retrieval_query="deploy rag-proxy",
        requested_model="demo-model",
        conversation_id="conv-1",
        trace_id="trace-1",
        hits=[
            ChunkHit(
                id="doc-1",
                text="retrieved secret chunk text",
                score=0.91234,
                source="dense",
                metadata={"title": "Deploy"},
            )
        ],
    )


def test_build_capture_records_writes_rag_and_finetune_streams(monkeypatch):
    """Successful turns should produce both training data and RAG improvement data."""
    monkeypatch.setattr(settings, "transcript_hit_preview_chars", 9)
    parsed = ParsedCompletion(
        assistant_message={"role": "assistant", "content": "Use systemd."},
        finish_reason="stop",
        usage={"total_tokens": 12},
        raw_ok=True,
    )

    rag_record = capture.build_rag_record(
        original_messages=[{"role": "user", "content": "How do I deploy rag-proxy?"}],
        ctx=_ctx(),
        parsed=parsed,
        path="v1/chat/completions",
        stream=False,
        timestamp="2026-01-01T00:00:00Z",
    )
    finetune_record = capture.build_finetune_record(
        original_messages=[{"role": "user", "content": "How do I deploy rag-proxy?"}],
        ctx=_ctx(),
        parsed=parsed,
        path="v1/chat/completions",
        stream=False,
        timestamp="2026-01-01T00:00:00Z",
    )

    assert rag_record["record_type"] == "rag_turn"
    assert rag_record["hits"][0]["text_preview"] == "retrieved"
    assert rag_record["qa_pair"] == {
        "question": "deploy rag-proxy",
        "answer": "Use systemd.",
    }
    assert finetune_record["record_type"] == "finetune_turn"
    assert finetune_record["assistant"] == {"role": "assistant", "content": "Use systemd."}
    assert finetune_record["usage"] == {"total_tokens": 12}


def test_build_finetune_record_skips_parse_failures_and_meta_prompts():
    """Fine-tuning output must exclude turns without a real user/assistant pair."""
    parsed = ParsedCompletion(parse_error="bad json")
    assert capture.build_finetune_record(
        original_messages=[],
        ctx=_ctx(),
        parsed=parsed,
        path="v1/chat/completions",
        stream=False,
    ) is None

    meta_ctx = _ctx()
    meta_ctx.query_text = "### Task:\nSuggest follow-up questions."
    parsed = ParsedCompletion(
        assistant_message={"role": "assistant", "content": "Question?"},
        raw_ok=True,
    )
    assert capture.build_finetune_record(
        original_messages=[],
        ctx=meta_ctx,
        parsed=parsed,
        path="v1/chat/completions",
        stream=False,
    ) is None


def test_capture_enabled_respects_header_opt_in_and_sampling(monkeypatch):
    """Operators can require explicit capture headers and sample traffic."""
    monkeypatch.setattr(settings, "enable_transcript_capture", True)
    monkeypatch.setattr(settings, "transcript_header_opt_in", True)
    monkeypatch.setattr(settings, "transcript_sample_rate", 1.0)

    assert not capture.capture_enabled({})
    assert capture.capture_enabled({"X-Capture-Log": "true"})

    monkeypatch.setattr(settings, "transcript_header_opt_in", False)
    monkeypatch.setattr(settings, "transcript_sample_rate", 0.0)
    assert not capture.capture_enabled({})
