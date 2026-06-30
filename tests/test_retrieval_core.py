"""Unit tests for shared retrieval request/response helpers."""

from rag_proxy.clients.retrieval_core import (
    EMBED_MODEL,
    dense_search_payload,
    embed_payload,
    parse_dense_hits,
    parse_embedding,
    parse_sparse_hits,
    prepare_embed_text,
)


def test_embed_payload_uses_nomic_model() -> None:
    payload = embed_payload("hello")
    assert payload == {"model": EMBED_MODEL, "input": "hello"}


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
