"""Sidecar API contract tests (offline; no torch or Qdrant)."""

from __future__ import annotations

import importlib.util
import sys
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from pathlib import Path

from rag_proxy.chunk_text import extract_chunk_text

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_sidecar_app(app_name: str, sidecar_dir: str):
    core = _load_module(f"{sidecar_dir}_core", f"sidecars/{sidecar_dir}/core.py")
    prior_core = sys.modules.get("core")
    sys.modules["core"] = core
    try:
        return _load_module(app_name, f"sidecars/{sidecar_dir}/app.py")
    finally:
        if prior_core is None:
            sys.modules.pop("core", None)
        else:
            sys.modules["core"] = prior_core


rerank_core = _load_module("rerank_core", "sidecars/rerank/core.py")
sparse_core = _load_module("sparse_core", "sidecars/sparse/core.py")


def test_rank_indices_orders_by_score():
    scores = [0.1, 0.9, 0.4]
    assert rerank_core.rank_indices(scores, top_k=2) == [1, 2]


def test_rank_indices_empty():
    assert rerank_core.rank_indices([], top_k=5) == []


def test_rank_indices_non_positive_top_k_returns_empty():
    assert rerank_core.rank_indices([0.1, 0.9], top_k=0) == []


def test_sparse_search_ranks_relevant_doc_first():
    filler = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor"
    registry = sparse_core.IndexRegistry()
    registry.rebuild(
        "test",
        [
            {
                "id": "infra",
                "payload": {
                    "text": (
                        "kubernetes homelab docker compose deployment "
                        "stack infrastructure server cluster guide"
                    ),
                },
            },
            {
                "id": "food",
                "payload": {
                    "text": "chocolate cake vanilla frosting baking recipe dessert kitchen",
                },
            },
            *[
                {"id": f"filler-{n}", "payload": {"text": f"{filler} document number {n}"}}
                for n in range(4)
            ],
        ],
    )
    assert registry.search("test", "kubernetes", 1)[0]["id"] == "infra"
    assert registry.search("test", "chocolate", 1)[0]["id"] == "food"


def test_sparse_search_unknown_collection_returns_empty():
    registry = sparse_core.IndexRegistry()
    registry.rebuild("loaded", [{"id": "1", "payload": {"text": "hello world document"}}])
    assert registry.search("other", "hello", 5) == []


def test_sparse_search_returns_match_in_single_doc_collection():
    """BM25 scores can be non-positive in tiny corpora; rank by overlap, not score sign."""
    registry = sparse_core.IndexRegistry()
    registry.rebuild("tiny", [{"id": "only", "payload": {"text": "kubernetes homelab guide"}}])
    results = registry.search("tiny", "kubernetes", 1)
    assert len(results) == 1
    assert results[0]["id"] == "only"


def test_sparse_search_skips_docs_without_query_token_overlap():
    registry = sparse_core.IndexRegistry()
    registry.rebuild(
        "demo",
        [
            {"id": "a", "payload": {"text": "alpha beta gamma"}},
            {"id": "b", "payload": {"text": "delta epsilon zeta"}},
        ],
    )
    results = registry.search("demo", "alpha", 5)
    assert [hit["id"] for hit in results] == ["a"]


def test_extract_chunk_text_matches_rag_proxy_field_order():
    payload = {"content": "from content field", "text": "from text field"}
    assert extract_chunk_text({"payload": payload}) == "from text field"


def test_sparse_sidecar_search_http():
    sparse_app = _load_sidecar_app("sparse_app", "sparse")
    sparse_app.registry.rebuild(
        "demo",
        [
            {
                "id": "infra",
                "payload": {"text": "kubernetes homelab docker compose deployment"},
            },
            {"id": "food", "payload": {"text": "chocolate cake baking recipe"}},
            *[
                {
                    "id": f"filler-{n}",
                    "payload": {"text": f"filler document number {n} lorem ipsum dolor sit amet"},
                }
                for n in range(4)
            ],
        ],
    )

    async def noop_sync(_collection: str) -> int:
        return sparse_app.registry.doc_count(_collection)

    with patch.object(sparse_app, "sync_collection", new=noop_sync):
        with TestClient(sparse_app.app) as client:
            resp = client.post(
                "/search",
                json={"query": "kubernetes homelab deployment", "limit": 2, "collection": "demo"},
            )

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["id"] == "infra"
    assert results[0]["payload"]["text"]


def test_rerank_sidecar_http():
    rerank_app = _load_sidecar_app("rerank_app", "rerank")
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [0.2, 0.9]

    with patch.object(rerank_app, "get_encoder", return_value=mock_encoder):
        with TestClient(rerank_app.app) as client:
            resp = client.post(
                "/rerank",
                json={
                    "pairs": [
                        {"query": "homelab deploy", "document": "weak match"},
                        {"query": "homelab deploy", "document": "strong homelab deploy match"},
                    ],
                    "top_k": 2,
                },
            )

    assert resp.status_code == 200
    assert resp.json()["indices"] == [1, 0]
    mock_encoder.predict.assert_called_once()
