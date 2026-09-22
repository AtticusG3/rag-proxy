"""Unit tests for shared retrieval request/response helpers."""

from rag_proxy.clients.retrieval_core import (
    canonical_point_id,
    dense_search_payload,
    embed_input_too_large,
    embed_payload,
    parse_dense_hits,
    parse_embedding,
    parse_sparse_hits,
    prepare_embed_text,
)
from rag_proxy.config import settings


def test_embed_payload_uses_settings_embed_model() -> None:
    """Embeddings must send settings.embed_model so cutover to embed-pool is env-driven."""
    payload = embed_payload("hello")
    assert payload == {"model": settings.embed_model, "input": "hello"}


def test_embed_payload_follows_monkeypatched_settings_model(monkeypatch) -> None:
    monkeypatch.setattr(settings, "embed_model", "embed-pool")
    assert embed_payload("hello")["model"] == "embed-pool"


def test_embed_payload_honors_explicit_model_override() -> None:
    payload = embed_payload("hello", model="embed-pool")
    assert payload == {"model": "embed-pool", "input": "hello"}


def test_embed_input_too_large_matches_llama_server_and_nomic() -> None:
    """Query embed must shrink on llama-server 400s, not only the old nomic phrase.

    A context overflow that is not retried fail-opens the whole RAG turn.
    """
    assert embed_input_too_large('{"error":"exceed_context_size"}')
    assert embed_input_too_large("request exceeds the available max context size")
    assert embed_input_too_large("input too large to process")
    assert not embed_input_too_large("model not found")


def test_canonical_point_id_folds_dashed_uuid_to_bare_hex() -> None:
    """Qdrant echoes dashed UUIDs for the bare hex ids TurboVec search returns."""
    bare = "00000468c899c290e5c51718ca7521d3"
    dashed = "00000468-c899-c290-e5c5-1718ca7521d3"
    assert canonical_point_id(dashed) == bare
    assert canonical_point_id(bare) == bare
    assert canonical_point_id("sparse-only") == "sparse-only"
    assert canonical_point_id(42) == "42"


def test_prepare_embed_text_tail_truncates() -> None:
    text = "x" * 100
    assert prepare_embed_text(text, 20) == "x" * 20


def test_dense_search_payload_omits_zero_threshold_when_requested() -> None:
    body = dense_search_payload([0.1], 5, 0.0, omit_zero_threshold=True)
    assert "score_threshold" not in body


def test_dense_search_payload_includes_positive_threshold() -> None:
    body = dense_search_payload([0.1], 5, 0.72, omit_zero_threshold=True)
    assert body["score_threshold"] == 0.72


def test_parse_embedding_extracts_vector() -> None:
    assert parse_embedding({"data": [{"embedding": [0.1, 0.2]}]}) == [0.1, 0.2]


def test_parse_dense_hits_returns_result_list() -> None:
    hits = parse_dense_hits({"result": [{"id": "1"}]})
    assert hits == [{"id": "1"}]


def test_parse_sparse_hits_returns_results_list() -> None:
    hits = parse_sparse_hits({"results": [{"id": "s1"}]})
    assert hits == [{"id": "s1"}]
