"""OpenAI-style /v1/rerank adapter mapping (pairs <-> indices)."""

from __future__ import annotations

from rag_proxy.clients.rerank_adapter import (
    parse_rerank_indices,
    pairs_to_openai_documents,
    rerank_request_payload,
    rerank_request_url,
    use_openai_rerank,
)
from rag_proxy.config import settings


def test_sidecar_rerank_url_and_payload_unchanged(monkeypatch) -> None:
    """Default RERANK_API=sidecar must keep the cognitive sidecar contract."""
    monkeypatch.setattr(settings, "rerank_api", "sidecar")
    pairs = [{"query": "q", "document": "a"}, {"query": "q", "document": "b"}]
    assert not use_openai_rerank()
    assert rerank_request_url("http://rerank.test") == "http://rerank.test/rerank"
    assert rerank_request_payload(pairs, 2) == {"pairs": pairs, "top_k": 2}
    assert parse_rerank_indices({"indices": [1, 0]}, n_pairs=2, top_k=2) == [1, 0]


def test_openai_rerank_maps_pairs_to_v1_body_and_results(monkeypatch) -> None:
    """RERANK_API=openai must speak llama-swap /v1/rerank so swap can replace the sidecar."""
    monkeypatch.setattr(settings, "rerank_api", "openai")
    monkeypatch.setattr(settings, "rerank_model", "reranker-pool")
    pairs = [
        {"query": "what is panda?", "document": "hi"},
        {"query": "what is panda?", "document": "a bear in China"},
    ]
    assert use_openai_rerank()
    assert rerank_request_url("http://127.0.0.1:8081/") == "http://127.0.0.1:8081/v1/rerank"
    assert pairs_to_openai_documents(pairs) == (
        "what is panda?",
        ["hi", "a bear in China"],
    )
    assert rerank_request_payload(pairs, 1) == {
        "model": "reranker-pool",
        "query": "what is panda?",
        "documents": ["hi", "a bear in China"],
        "top_n": 1,
    }
    indices = parse_rerank_indices(
        {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.1},
            ]
        },
        n_pairs=2,
        top_k=1,
    )
    assert indices == [1]
