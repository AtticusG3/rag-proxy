"""SIDECAR_TOKEN opt-in gate on sidecar work-doing routes (#56).

Each sidecar is replayed in both configurations: token unset (behaves exactly
as before the gate) and token set (401 on missing or wrong header before any
work, read-only routes unchanged). Modules are loaded fresh per case because
the sidecar apps read the env at import time.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_sidecar_app(name: str, sidecar_dir: str):
    core = _load_module(f"{name}_core", f"sidecars/{sidecar_dir}/core.py")
    prior_core = sys.modules.get("core")
    sys.modules["core"] = core
    try:
        return _load_module(name, f"sidecars/{sidecar_dir}/app.py")
    finally:
        if prior_core is None:
            sys.modules.pop("core", None)
        else:
            sys.modules["core"] = prior_core


def _load_rerank_app(name: str, monkeypatch, token: str | None):
    if token is None:
        monkeypatch.delenv("SIDECAR_TOKEN", raising=False)
    else:
        monkeypatch.setenv("SIDECAR_TOKEN", token)
    return _load_sidecar_app(name, "rerank")


def _load_sparse_app(name: str, monkeypatch, token: str | None):
    if token is None:
        monkeypatch.delenv("SIDECAR_TOKEN", raising=False)
    else:
        monkeypatch.setenv("SIDECAR_TOKEN", token)
    return _load_sidecar_app(name, "sparse")


def _load_turbovec_app(name: str, monkeypatch, token: str | None):
    if token is None:
        monkeypatch.delenv("SIDECAR_TOKEN", raising=False)
    else:
        monkeypatch.setenv("SIDECAR_TOKEN", token)
    return _load_sidecar_app(name, "turbovec")


# --- rerank: POST /rerank runs the model (the work-doing route) ---


def _rerank_body() -> dict:
    return {
        "pairs": [
            {"query": "homelab deploy", "document": "weak match"},
            {"query": "homelab deploy", "document": "strong homelab deploy match"},
        ],
        "top_k": 2,
    }


@pytest.mark.parametrize("token", [None, ""])
def test_rerank_token_unset_behaves_as_before(monkeypatch, token):
    rerank_app = _load_rerank_app("rerank_app_unset", monkeypatch, token)
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [0.2, 0.9]
    with patch.object(rerank_app, "get_encoder", return_value=mock_encoder):
        with TestClient(rerank_app.app) as client:
            resp = client.post("/rerank", json=_rerank_body())
    assert resp.status_code == 200
    assert resp.json()["indices"] == [1, 0]


def test_rerank_rejects_missing_or_wrong_header_when_token_set(monkeypatch):
    rerank_app = _load_rerank_app("rerank_app_set", monkeypatch, "s3cret")
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [0.2, 0.9]
    with patch.object(rerank_app, "get_encoder", return_value=mock_encoder):
        with TestClient(rerank_app.app) as client:
            missing = client.post("/rerank", json=_rerank_body())
            wrong = client.post(
                "/rerank", json=_rerank_body(), headers={"X-Internal-Token": "nope"}
            )
            good = client.post(
                "/rerank", json=_rerank_body(), headers={"X-Internal-Token": "s3cret"}
            )
            health = client.get("/health")
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert good.status_code == 200
    assert health.status_code == 200
    mock_encoder.predict.assert_called_once()


# --- sparse: POST /reindex rebuilds from Qdrant (mutating) ---


def _patch_sparse_sync(sparse_app, docs: int = 3):
    async def noop_sync(_collection: str) -> int:
        return docs

    return patch.object(sparse_app, "sync_collection", new=noop_sync)


@pytest.mark.parametrize("token", [None, ""])
def test_sparse_reindex_unset_behaves_as_before(monkeypatch, token):
    sparse_app = _load_sparse_app("sparse_app_unset", monkeypatch, token)
    with _patch_sparse_sync(sparse_app):
        with TestClient(sparse_app.app) as client:
            resp = client.post("/reindex", json={"collection": "demo"})
    assert resp.status_code == 200
    assert resp.json()["docs"] == 3


def test_sparse_reindex_rejects_missing_or_wrong_header_when_token_set(monkeypatch):
    sparse_app = _load_sparse_app("sparse_app_set", monkeypatch, "s3cret")
    with _patch_sparse_sync(sparse_app):
        with TestClient(sparse_app.app) as client:
            missing = client.post("/reindex", json={"collection": "demo"})
            wrong = client.post(
                "/reindex",
                json={"collection": "demo"},
                headers={"X-Internal-Token": "nope"},
            )
            good = client.post(
                "/reindex",
                json={"collection": "demo"},
                headers={"X-Internal-Token": "s3cret"},
            )
            search = client.post("/search", json={"query": "hello", "collection": "demo"})
            health = client.get("/health")
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert good.status_code == 200
    # Read-only routes never require the header.
    assert search.status_code == 200
    assert health.status_code == 200


# --- turbovec: /add /remove /reindex /save mutate the dense index ---


def _turbovec_fake_index():
    idx = MagicMock()
    idx.stats.return_value = {"vectors": 0}
    idx.add.return_value = 1
    idx.remove.return_value = 1
    idx.search.return_value = []
    return idx


@pytest.mark.parametrize(
    "path,payload",
    [("/add", {"ids": ["a"], "vectors": [[0.0]]}), ("/remove", {"ids": ["a"]})],
)
def test_turbovec_unset_behaves_as_before(monkeypatch, path, payload):
    tv_app = _load_turbovec_app("turbovec_app_unset", monkeypatch, None)
    monkeypatch.setattr(tv_app, "index", _turbovec_fake_index())
    # No context manager: skip lifespan (the real index load needs the turbovec
    # package). The gate is a route dependency, so it runs regardless.
    client = TestClient(tv_app.app)
    with patch.object(tv_app, "sync_from_qdrant", new=AsyncMock(return_value=0)):
        resp = client.post(path, json=payload)
    assert resp.status_code == 200


def test_turbovec_mutating_routes_reject_when_token_set(monkeypatch):
    tv_app = _load_turbovec_app("turbovec_app_set", monkeypatch, "s3cret")
    idx = _turbovec_fake_index()
    monkeypatch.setattr(tv_app, "index", idx)
    calls = [
        ("/add", {"ids": ["a"], "vectors": [[0.0]]}),
        ("/remove", {"ids": ["a"]}),
        ("/reindex", {"collection": "demo"}),
        ("/save", None),
    ]
    client = TestClient(tv_app.app)
    with patch.object(tv_app, "sync_from_qdrant", new=AsyncMock(return_value=0)):
        for path, payload in calls:
            missing = client.post(path, json=payload or {})
            wrong = client.post(
                path, json=payload or {}, headers={"X-Internal-Token": "nope"}
            )
            good = client.post(
                path, json=payload or {}, headers={"X-Internal-Token": "s3cret"}
            )
            assert missing.status_code == 401, path
            assert wrong.status_code == 401, path
            assert good.status_code == 200, path
        # Read-only routes never require the header.
        health = client.get("/health")
        search = client.post("/search", json={"vector": [0.0], "limit": 1})
    assert health.status_code == 200
    assert search.status_code == 200


# --- clients send X-Internal-Token when SIDECAR_TOKEN is configured ---


class _RecordingClient:
    """httpx.Client stand-in capturing the kwargs of every post()."""

    def __init__(self, posts: list[dict], response: MagicMock | None = None):
        self.posts = posts
        self.response = response or _ok_response()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, url, **kwargs):
        self.posts.append({"url": url, **kwargs})
        return self.response


def _ok_response(body: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=body if body is not None else {})
    return resp


def _turbovec_client_module(monkeypatch, token: str | None):
    if token is None:
        monkeypatch.delenv("SIDECAR_TOKEN", raising=False)
    else:
        monkeypatch.setenv("SIDECAR_TOKEN", token)
    import ingest.turbovec_client as tv

    return tv


def test_ingest_turbovec_add_sends_token_header_when_configured(monkeypatch):
    tv = _turbovec_client_module(monkeypatch, "s3cret")
    posts: list[dict] = []
    monkeypatch.setattr(tv.httpx, "Client", lambda **kw: _RecordingClient(posts))
    added = tv.add_points("http://tv.test", [{"id": "abc", "vector": [0.1, 0.2]}])
    assert added == 1 or added is None
    assert posts[0]["headers"]["X-Internal-Token"] == "s3cret"


def test_ingest_turbovec_remove_and_reindex_sends_token_header_when_configured(monkeypatch):
    tv = _turbovec_client_module(monkeypatch, "s3cret")
    posts: list[dict] = []
    monkeypatch.setattr(tv.httpx, "Client", lambda **kw: _RecordingClient(posts))
    tv.remove_ids("http://tv.test", ["abc"])
    tv.trigger_reindex("http://tv.test", "coll")
    assert [p["url"] for p in posts] == [
        "http://tv.test/remove",
        "http://tv.test/reindex",
    ]
    assert all(p["headers"]["X-Internal-Token"] == "s3cret" for p in posts)


@pytest.mark.parametrize("token", [None, ""])
def test_ingest_turbovec_sends_no_token_header_when_unset(monkeypatch, token):
    tv = _turbovec_client_module(monkeypatch, token)
    posts: list[dict] = []
    monkeypatch.setattr(tv.httpx, "Client", lambda **kw: _RecordingClient(posts))
    tv.remove_ids("http://tv.test", ["abc"])
    assert posts[0]["headers"] == {}


def test_ingest_sparse_reindex_sends_token_header_when_configured(monkeypatch):
    monkeypatch.setenv("SIDECAR_TOKEN", "s3cret")
    from ingest import worker

    posts: list[dict] = []
    monkeypatch.setattr(
        worker.httpx,
        "Client",
        lambda **kw: _RecordingClient(posts, _ok_response({"docs": 5})),
    )
    config = worker.IngestConfig(
        zim_dir="/tmp/zim",
        upload_dir="/tmp/up",
        embed_url="http://embed.test",
        qdrant_url="http://qdrant.test",
        qdrant_collection="coll",
        sparse_index_url="http://sparse.test",
    )
    assert worker.trigger_sparse_reindex(config) == 5
    assert posts[0]["headers"]["X-Internal-Token"] == "s3cret"


def test_migrate_script_reindex_and_save_send_token_header(monkeypatch):
    monkeypatch.setenv("SIDECAR_TOKEN", "s3cret")
    import scripts.migrate_qdrant_sidecars as mig

    posts: list[dict] = []
    monkeypatch.setattr(mig.httpx, "Client", lambda **kw: _RecordingClient(posts))
    client = mig.httpx.Client()
    mig._post_reindex("http://tv.test", "coll", client)
    mig._post_save("http://tv.test", client)
    assert all(p["headers"]["X-Internal-Token"] == "s3cret" for p in posts)


def test_proxy_rerank_stage_sends_token_header_when_configured(monkeypatch):
    from conftest import FakeAsyncClient
    from rag_proxy.config import settings
    from rag_proxy.context import ChunkHit, RequestContext
    from rag_proxy.stages import tier2_rerank

    monkeypatch.setattr(settings, "enable_reranker", True)
    monkeypatch.setattr(settings, "reranker_url", "http://rerank.test")
    monkeypatch.setattr(settings, "sidecar_token", "s3cret")
    monkeypatch.setattr(settings, "rerank_api", "sidecar")

    ctx = RequestContext(
        query_text="q",
        retrieval_query="q",
        hits=[ChunkHit(id="a", text="doc one", score=0.9)],
    )
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"indices": [0]})
    post = AsyncMock(return_value=resp)
    with patch.object(
        tier2_rerank, "get_reranker_client", return_value=FakeAsyncClient(post)
    ):
        import asyncio

        asyncio.run(tier2_rerank.run_rerank(ctx))
    assert post.await_args.kwargs["headers"]["X-Internal-Token"] == "s3cret"


def test_proxy_rerank_stage_sends_no_token_header_when_unset(monkeypatch):
    from conftest import FakeAsyncClient
    from rag_proxy.config import settings
    from rag_proxy.context import ChunkHit, RequestContext
    from rag_proxy.stages import tier2_rerank

    monkeypatch.setattr(settings, "enable_reranker", True)
    monkeypatch.setattr(settings, "reranker_url", "http://rerank.test")
    monkeypatch.setattr(settings, "sidecar_token", "")

    ctx = RequestContext(
        query_text="q",
        retrieval_query="q",
        hits=[ChunkHit(id="a", text="doc one", score=0.9)],
    )
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"indices": [0]})
    post = AsyncMock(return_value=resp)
    with patch.object(
        tier2_rerank, "get_reranker_client", return_value=FakeAsyncClient(post)
    ):
        import asyncio

        asyncio.run(tier2_rerank.run_rerank(ctx))
    assert post.await_args.kwargs["headers"] == {}


def test_retrieve_sync_rerank_sends_token_header_but_embed_never(monkeypatch):
    """The sidecar secret goes to the reranker only, never to embed or Qdrant."""
    from rag_proxy.clients import retrieve_sync as rs

    config = rs.RetrieveConfig(
        embed_url="http://embed.test",
        qdrant_url="http://qdrant.test",
        qdrant_collection="coll",
        sparse_index_url="",
        reranker_url="http://rerank.test",
        similarity_threshold=0.5,
        hybrid_dense_weight=0.7,
        embed_max_chars=100,
        enable_hybrid=False,
        enable_rerank=True,
        sidecar_token="s3cret",
    )
    posts: list[dict] = []

    class FakeClient:
        def __init__(self, *args, headers=None, **kwargs):
            self.headers = headers or {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, **kwargs):
            posts.append({"url": url, "headers": dict(self.headers)})
            return _ok_response({"indices": [0]})

    with patch.object(rs.httpx, "Client", FakeClient):
        order = rs.rerank_pairs(
            config,
            [{"query": "q", "document": "doc"}],
            top_k=1,
        )
    assert order == [0]
    assert posts[0]["url"].endswith("/rerank")
    assert posts[0]["headers"]["X-Internal-Token"] == "s3cret"

    embed_posts: list[dict] = []

    class EmbedClient(FakeClient):
        def post(self, url, **kwargs):
            embed_posts.append({"url": url, "headers": dict(self.headers)})
            return _ok_response({"data": [{"embedding": [0.1]}]})

    with patch.object(rs.httpx, "Client", EmbedClient):
        rs.embed_query(config, "hello")
    assert embed_posts
    assert "X-Internal-Token" not in embed_posts[0]["headers"]


def test_openai_rerank_mode_sends_no_sidecar_token(monkeypatch):
    """RERANK_API=openai targets llama-swap, not our sidecar; no secret leaks."""
    from rag_proxy.clients.rerank_adapter import rerank_auth_headers
    from rag_proxy.config import settings

    monkeypatch.setattr(settings, "rerank_api", "openai")
    assert rerank_auth_headers("s3cret") == {}
    monkeypatch.setattr(settings, "rerank_api", "sidecar")
    assert rerank_auth_headers("s3cret") == {"X-Internal-Token": "s3cret"}
    assert rerank_auth_headers("") == {}


def test_sidecar_token_documented_in_env_example():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"(?m)^SIDECAR_TOKEN=\s*$", text)


def test_mcp_rag_config_reads_sidecar_token(monkeypatch):
    sys.path.insert(0, str(ROOT / "sidecars" / "mcp_rag"))
    import retrieve as mcp_retrieve

    monkeypatch.setenv("SIDECAR_TOKEN", " s3cret ")
    settings = mcp_retrieve.RetrieveSettings.from_env()
    assert settings.sidecar_token == "s3cret"
    config = mcp_retrieve._to_retrieve_config(settings)
    assert config.sidecar_token == "s3cret"
