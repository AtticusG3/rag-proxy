"""TurboVec id mapping, merge helpers, and optional sidecar contract tests."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


turbovec_core = _load_module("turbovec_core_test", "sidecars/turbovec/core.py")


def test_hex_id_to_u64_uses_first_16_hex_chars():
    hid = "abcdef0123456789deadbeefcafebabe"
    assert turbovec_core.hex_id_to_u64(hid) == int("abcdef0123456789", 16)


def test_hex_id_to_u64_rejects_short_ids():
    with pytest.raises(ValueError):
        turbovec_core.hex_id_to_u64("abc")


def test_normalize_qdrant_hex_id_rejects_short_ids():
    with pytest.raises(ValueError, match="32 chars"):
        turbovec_core.normalize_qdrant_hex_id("abc")


def test_merge_dense_scores_with_payloads_preserves_order():
    from rag_proxy.clients.retrieval_core import merge_dense_scores_with_payloads

    scored = [
        {"id": "a" * 32, "score": 0.9},
        {"id": "b" * 32, "score": 0.5},
    ]
    points = [
        {"id": "b" * 32, "payload": {"text": "second"}},
        {"id": "a" * 32, "payload": {"text": "first"}},
    ]
    merged = merge_dense_scores_with_payloads(scored, points)
    assert [m["id"] for m in merged] == ["a" * 32, "b" * 32]
    assert merged[0]["payload"]["text"] == "first"
    assert merged[1]["score"] == 0.5


def test_search_turbovec_dense_fail_open_without_url(monkeypatch):
    from rag_proxy.clients import retrieval_async as ra

    monkeypatch.setattr(ra.settings, "dense_backend", "turbovec")
    monkeypatch.setattr(ra.settings, "turbovec_url", "")
    hits = asyncio.run(ra.search_turbovec_dense([0.1] * 8, limit=3))
    assert hits == []


def test_search_qdrant_dense_routes_to_turbovec(monkeypatch):
    from rag_proxy.clients import retrieval_async as ra

    monkeypatch.setattr(ra.settings, "dense_backend", "turbovec")
    monkeypatch.setattr(ra.settings, "turbovec_url", "http://127.0.0.1:8097")

    async def fake_tv(vector, limit=None, score_threshold=None):
        return [{"id": "x", "score": 1.0, "payload": {"text": "ok"}}]

    monkeypatch.setattr(ra, "search_turbovec_dense", fake_tv)
    hits = asyncio.run(ra.search_qdrant_dense([0.0] * 8))
    assert hits[0]["payload"]["text"] == "ok"


def test_search_turbovec_merges_payloads(monkeypatch):
    from rag_proxy.clients import retrieval_async as ra

    monkeypatch.setattr(ra.settings, "dense_backend", "turbovec")
    monkeypatch.setattr(ra.settings, "turbovec_url", "http://tv")
    monkeypatch.setattr(ra.settings, "qdrant_url", "http://qd")
    monkeypatch.setattr(ra.settings, "qdrant_collection", "col")
    monkeypatch.setattr(ra.settings, "top_k", 5)
    monkeypatch.setattr(ra.settings, "similarity_threshold", 0.0)

    mock_tv = MagicMock()
    mock_tv.post = AsyncMock(
        return_value=MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(
                return_value={"results": [{"id": "aa" * 16, "score": 0.8}]}
            ),
        )
    )
    mock_qd = MagicMock()
    mock_qd.post = AsyncMock(
        return_value=MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(
                return_value={
                    "result": [{"id": "aa" * 16, "payload": {"text": "chunk"}}]
                }
            ),
        )
    )
    monkeypatch.setattr(ra, "get_turbovec_client", lambda: mock_tv)
    monkeypatch.setattr(ra, "get_qdrant_client", lambda: mock_qd)

    hits = asyncio.run(
        ra.search_turbovec_dense([0.1] * 8, limit=3, score_threshold=0.0)
    )
    assert len(hits) == 1
    assert hits[0]["id"] == "aa" * 16
    assert hits[0]["payload"]["text"] == "chunk"
    assert hits[0]["score"] == 0.8


def test_add_points_fail_open_returns_none_when_sidecar_errors():
    """TurboVec dual-write must not abort ingest when /add fails."""
    from ingest.turbovec_client import add_points

    mock_client = MagicMock()
    mock_client.post.side_effect = RuntimeError("turbovec sidecar offline")
    point_id = "a" * 32
    points = [{"id": point_id, "vector": [0.1, 0.2, 0.3, 0.4]}]

    assert add_points("http://127.0.0.1:8097", points, client=mock_client) is None


def test_upsert_batch_completes_when_qdrant_ok_and_turbovec_add_fails(monkeypatch):
    """Ingest batch count is returned even if TurboVec add fails after Qdrant upsert."""
    from ingest import pipeline as pl

    qdrant_upserted: list[int] = []

    def fake_upsert(_url, _collection, points, **kwargs):
        qdrant_upserted.append(len(points))

    monkeypatch.setattr(pl, "upsert_points", fake_upsert)

    mock_tv = MagicMock()
    mock_tv.post.side_effect = httpx.HTTPError("connection refused")

    batch = [("Doc", "corpus/sample.txt", "chunk body")]
    embeddings = [[0.0] * 768]

    write_ctx = pl.IngestWriteContext(
        qdrant_url="http://qdrant",
        collection="test_col",
        qdrant_client=MagicMock(),
        turbovec_url="http://127.0.0.1:8097",
        turbovec_client=mock_tv,
    )
    count = pl._upsert_batch(
        batch,
        embeddings,
        0,
        write_ctx=write_ctx,
    )

    assert count == 1
    assert qdrant_upserted == [1]
    mock_tv.post.assert_called_once()


def test_delete_source_points_lists_ids_qdrant_delete_then_remove(monkeypatch):
    """TurboVec remove runs only after Qdrant filter delete succeeds."""
    from ingest.dual_write import delete_source_points

    order: list[object] = []
    listed_ids = [
        "11111111111111111111111111111111",
        "22222222222222222222222222222222",
    ]

    def fake_list(*_args, **_kwargs):
        order.append("list_ids")
        return list(listed_ids)

    def fake_qdrant_delete(*_args, **_kwargs):
        order.append("qdrant_delete")

    def fake_remove(_url, ids, client=None):
        order.append(("remove_ids", list(ids)))
        return len(ids)

    monkeypatch.setattr(
        "ingest.dual_write.list_point_ids_by_source", fake_list
    )
    monkeypatch.setattr("ingest.dual_write.delete_by_source", fake_qdrant_delete)
    monkeypatch.setattr("ingest.turbovec_client.remove_ids", fake_remove)

    delete_source_points(
        "http://qdrant",
        "col",
        "corpus/sample.zim",
        turbovec_url="http://127.0.0.1:8097",
    )

    assert order[0] == "list_ids"
    assert order[1] == "qdrant_delete"
    assert order[2] == ("remove_ids", listed_ids)


def test_delete_source_points_skips_turbovec_when_qdrant_delete_raises(
    monkeypatch,
):
    """TurboVec /remove must not run if Qdrant delete fails."""
    from ingest.dual_write import delete_source_points

    monkeypatch.setattr(
        "ingest.dual_write.list_point_ids_by_source",
        lambda *_a, **_k: ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
    )

    def fail_qdrant(*_args, **_kwargs):
        raise RuntimeError("qdrant unavailable")

    remove_called: list[bool] = []

    def fake_remove(*_a, **_k):
        remove_called.append(True)
        return 1

    monkeypatch.setattr("ingest.dual_write.delete_by_source", fail_qdrant)
    monkeypatch.setattr("ingest.turbovec_client.remove_ids", fake_remove)

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        delete_source_points(
            "http://qdrant",
            "col",
            "orphan.zim",
            turbovec_url="http://127.0.0.1:8097",
        )

    assert remove_called == []


def test_delete_source_points_qdrant_deleted_when_remove_ids_returns_none(
    monkeypatch,
):
    """Qdrant delete completes even when TurboVec remove fails open (returns None)."""
    from ingest.dual_write import delete_source_points

    monkeypatch.setattr(
        "ingest.dual_write.list_point_ids_by_source",
        lambda *_a, **_k: ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
    )
    qdrant_delete_calls: list[str] = []

    def track_qdrant(*_a, **_k):
        qdrant_delete_calls.append("ok")

    monkeypatch.setattr("ingest.dual_write.delete_by_source", track_qdrant)
    monkeypatch.setattr(
        "ingest.turbovec_client.remove_ids",
        lambda *_a, **_k: None,
    )

    delete_source_points(
        "http://qdrant",
        "col",
        "orphan.zim",
        turbovec_url="http://127.0.0.1:8097",
    )

    assert qdrant_delete_calls == ["ok"]


def test_sync_dense_search_turbovec_then_qdrant_retrieve(monkeypatch):
    """Sync retrieval with dense_backend=turbovec searches TurboVec then hydrates from Qdrant."""
    from rag_proxy.clients import retrieve_sync as rs

    point_id = "aa" * 16
    post_sequence: list[str] = []

    class SyncClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, json=None):
            post_sequence.append(url)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if url.rstrip("/").endswith("/search"):
                resp.json.return_value = {
                    "results": [{"id": point_id, "score": 0.88}]
                }
            else:
                resp.json.return_value = {
                    "result": [
                        {"id": point_id, "payload": {"text": "hydrated chunk"}}
                    ]
                }
            return resp

    monkeypatch.setattr(rs.httpx, "Client", SyncClient)

    config = rs.RetrieveConfig(
        embed_url="http://embed",
        qdrant_url="http://qdrant",
        qdrant_collection="col",
        sparse_index_url="",
        reranker_url="",
        similarity_threshold=0.0,
        hybrid_dense_weight=0.5,
        embed_max_chars=2000,
        dense_backend="turbovec",
        turbovec_url="http://turbovec",
    )

    hits = rs.dense_search(config, [0.1] * 8, limit=3)

    assert len(post_sequence) == 2
    assert post_sequence[0] == "http://turbovec/search"
    assert post_sequence[1] == "http://qdrant/collections/col/points"
    assert len(hits) == 1
    assert hits[0]["id"] == point_id
    assert hits[0]["score"] == 0.88
    assert hits[0]["payload"]["text"] == "hydrated chunk"


def test_vectors_config_on_disk(monkeypatch):
    from ingest import qdrant_writer as qw

    monkeypatch.delenv("QDRANT_VECTORS_ON_DISK", raising=False)
    assert qw._vectors_config(768) == {"size": 768, "distance": "Cosine"}
    monkeypatch.setenv("QDRANT_VECTORS_ON_DISK", "true")
    assert qw._vectors_config(768)["on_disk"] is True


@pytest.mark.skipif(
    not turbovec_core.HAS_TURBOVEC,
    reason="turbovec package not installed",
)
def test_turbovec_index_add_search_remove_roundtrip():
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.tvim"
        idx = turbovec_core.TurboIndex(dim=8, bit_width=4, index_path=path)
        try:
            ids = [
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ]
            vectors = np.eye(2, 8, dtype=np.float32).tolist()
            assert idx.add(ids, vectors) == 2
            results = idx.search(vectors[0], limit=1)
            assert results[0]["id"] == ids[0]
            assert idx.remove([ids[0]]) == 1
            assert len(idx) == 1
            idx.save()
            assert path.is_file()
        finally:
            idx.close()


@pytest.mark.skipif(
    not turbovec_core.HAS_TURBOVEC,
    reason="turbovec package not installed",
)
def test_rebuild_keeps_old_index_searchable_until_commit():
    """A reindex must not blank the sidecar while it scrolls Qdrant.

    Resetting in place makes every dense search return nothing for the length of
    the rebuild, which silently drops retrieval when DENSE_BACKEND=turbovec.
    """
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.tvim"
        idx = turbovec_core.TurboIndex(dim=8, bit_width=4, index_path=path)
        try:
            old_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            new_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            vectors = np.eye(2, 8, dtype=np.float32).tolist()
            idx.add([old_id], [vectors[0]])

            rebuild = idx.rebuild()
            rebuild.add([new_id], [vectors[1]])
            assert idx.search(vectors[0], limit=1)[0]["id"] == old_id

            assert rebuild.commit() == 1
            assert idx.search(vectors[1], limit=1)[0]["id"] == new_id
            assert idx.search(vectors[0], limit=1, score_threshold=0.99) == []
        finally:
            idx.close()


@pytest.mark.skipif(
    not turbovec_core.HAS_TURBOVEC,
    reason="turbovec package not installed",
)
def test_aborted_rebuild_leaves_live_index_intact():
    """A failed scroll must not strand the sidecar on a half-built corpus."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.tvim"
        idx = turbovec_core.TurboIndex(dim=8, bit_width=4, index_path=path)
        try:
            old_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            vectors = np.eye(2, 8, dtype=np.float32).tolist()
            idx.add([old_id], [vectors[0]])

            rebuild = idx.rebuild()
            rebuild.add(["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"], [vectors[1]])
            rebuild.abort()

            assert len(idx) == 1
            assert idx.search(vectors[0], limit=1)[0]["id"] == old_id
        finally:
            idx.close()


@pytest.mark.skipif(
    not turbovec_core.HAS_TURBOVEC,
    reason="turbovec package not installed",
)
def test_search_skips_ann_hit_when_id_map_row_missing():
    """ANN may return a u64, but without id_map we must not synthesize a Qdrant id."""
    import sqlite3

    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.tvim"
        idx = turbovec_core.TurboIndex(dim=8, bit_width=4, index_path=path)
        try:
            point_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            assert idx.add([point_id], [vector]) == 1
            assert idx.search(vector, limit=1)[0]["id"] == point_id

            uid = turbovec_core.hex_id_to_u64(point_id)
            id_db = path.with_suffix(".ids.sqlite")
            conn = sqlite3.connect(str(id_db))
            conn.execute("DELETE FROM id_map WHERE u64 = ?", (str(uid),))
            conn.commit()
            conn.close()

            results = idx.search(vector, limit=1, score_threshold=0.0)
            assert results == []
        finally:
            idx.close()


@pytest.mark.skipif(
    not turbovec_core.HAS_TURBOVEC,
    reason="turbovec package not installed",
)
def test_search_score_threshold_filters_after_top_k():
    """score_threshold trims the top-k shortlist; limit=3 can return fewer than three hits."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.tvim"
        idx = turbovec_core.TurboIndex(dim=8, bit_width=4, index_path=path)
        try:
            ids = [
                "11111111111111111111111111111111",
                "22222222222222222222222222222222",
                "33333333333333333333333333333333",
                "44444444444444444444444444444444",
            ]
            vectors = np.eye(4, 8, dtype=np.float32).tolist()
            idx.add(ids, vectors)
            query = vectors[0]
            top3 = idx.search(query, limit=3, score_threshold=None)
            assert len(top3) == 3
            weakest = min(hit["score"] for hit in top3)
            filtered = idx.search(
                query,
                limit=3,
                score_threshold=weakest + 1e-6,
            )
            assert len(filtered) < 3
            assert all(hit["score"] >= weakest + 1e-6 for hit in filtered)
        finally:
            idx.close()


@pytest.mark.skipif(
    not turbovec_core.HAS_TURBOVEC,
    reason="turbovec package not installed",
)
def test_add_rejects_non_32_char_hex_ids():
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.tvim"
        idx = turbovec_core.TurboIndex(dim=8, bit_width=4, index_path=path)
        try:
            with pytest.raises(ValueError, match="32 chars"):
                idx.add(["abc"], np.zeros((1, 8), dtype=np.float32).tolist())
        finally:
            idx.close()


@pytest.mark.skipif(
    not turbovec_core.HAS_TURBOVEC,
    reason="turbovec package not installed",
)
def test_turbovec_sidecar_http_add_search(monkeypatch):
    import numpy as np
    from fastapi.testclient import TestClient

    prior_core = sys.modules.get("core")
    sys.modules["core"] = turbovec_core
    try:
        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "idx.tvim"
            monkeypatch.setenv("TURBOVEC_DIM", "8")
            monkeypatch.setenv("TURBOVEC_BIT_WIDTH", "4")
            monkeypatch.setenv("TURBOVEC_INDEX_PATH", str(index_path))
            monkeypatch.setenv("TURBOVEC_AUTO_SAVE", "false")

            sys.modules.pop("turbovec_app_http", None)
            app_mod = _load_module("turbovec_app_http", "sidecars/turbovec/app.py")
            app_mod.DIM = 8
            app_mod.BIT_WIDTH = 4
            app_mod.INDEX_PATH = index_path
            app_mod.AUTO_SAVE = False

            with TestClient(app_mod.app) as client:
                health = client.get("/health")
                assert health.status_code == 200
                assert health.json()["status"] == "ok"

                vectors = np.eye(2, 8, dtype=np.float32).tolist()
                ids = [
                    "11111111111111111111111111111111",
                    "22222222222222222222222222222222",
                ]
                add = client.post("/add", json={"ids": ids, "vectors": vectors})
                assert add.status_code == 200
                assert add.json()["added"] == 2

                search = client.post(
                    "/search",
                    json={"vector": vectors[0], "limit": 1},
                )
                assert search.status_code == 200
                assert search.json()["results"][0]["id"] == ids[0]

                rem = client.post("/remove", json={"ids": [ids[0]]})
                assert rem.status_code == 200
                assert rem.json()["removed"] == 1
    finally:
        if prior_core is None:
            sys.modules.pop("core", None)
        else:
            sys.modules["core"] = prior_core
