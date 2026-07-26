"""Tests for Qdrant → sidecar migration helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ingest.sidecar_migrate import (
    health_count,
    health_max_points,
    needs_sidecar_rebuild,
    sparse_target_docs,
)
from rag_admin.db import AdminDatabase
from rag_admin.job_runner import JOB_SIDECAR_MIGRATE, BackgroundJobRunner


def test_sparse_target_respects_max_points_cap() -> None:
    assert sparse_target_docs(1_000_000, max_points=150_000) == 150_000
    assert sparse_target_docs(1_000_000, max_points=0) == 1_000_000
    assert sparse_target_docs(1_000_000, max_points=None) == 1_000_000
    assert sparse_target_docs(10, max_points=150_000) == 10


def test_needs_sidecar_rebuild_safe_when_caught_up_or_empty() -> None:
    assert needs_sidecar_rebuild(0, 0) is False
    assert needs_sidecar_rebuild(100, 100) is False
    assert needs_sidecar_rebuild(150_000, 150_000) is False
    assert needs_sidecar_rebuild(0, 100) is True
    assert needs_sidecar_rebuild(99, 100) is True


def test_needs_sidecar_rebuild_when_sidecar_holds_ghosts() -> None:
    """A sidecar ahead of Qdrant is serving deleted content, not 'caught up'.

    Clearing the collection leaves BM25/TurboVec full of entries whose points no
    longer exist, so retrieval keeps citing wiped documents until a rebuild.
    """
    assert needs_sidecar_rebuild(100, 0) is True
    assert needs_sidecar_rebuild(500, 100) is True


def test_health_helpers() -> None:
    assert health_count({"vectors": 12}, "vectors", "docs") == 12
    assert health_count({"docs": "9"}, "vectors", "docs") == 9
    assert health_count(None, "docs") == 0
    assert health_max_points({"max_points": 150000}) == 150000
    assert health_max_points({"max_points": 0}) == 0
    assert health_max_points({}) is None


@patch("rag_admin.job_runner.threading.Thread")
@patch("rag_admin.job_runner.subprocess.Popen")
def test_start_sidecar_migrate_registers_job(
    mock_popen: MagicMock, mock_thread: MagicMock, tmp_path: Path
) -> None:
    mock_proc = MagicMock()
    mock_proc.pid = 4242
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc
    mock_thread.return_value.start = MagicMock()

    db = AdminDatabase(str(tmp_path / "admin.sqlite"))
    runner = BackgroundJobRunner(
        db,
        repo_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
    )
    job_id = runner.start_sidecar_migrate(
        {
            "qdrant_url": "http://127.0.0.1:6333",
            "collection": "nomad_knowledge_base",
            "turbovec_url": "http://127.0.0.1:18097",
            "sparse_url": "http://127.0.0.1:18096",
        }
    )
    assert job_id
    active = runner.active_job(JOB_SIDECAR_MIGRATE)
    assert active is not None
    cmd = mock_popen.call_args[0][0]
    assert any("migrate_qdrant_sidecars.py" in part for part in cmd)


def test_run_migration_skips_when_already_synced(monkeypatch, capsys) -> None:
    """Re-clicking migrate must exit 0 and report skips, not fail."""
    from scripts import migrate_qdrant_sidecars as mig

    class FakeResp:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError("http error")

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url: str):
            if "/collections/" in url:
                return FakeResp({"result": {"points_count": 100}})
            if url.rstrip("/").endswith("/health"):
                if "18097" in url:
                    return FakeResp({"status": "ok", "vectors": 100})
                return FakeResp({"status": "ok", "docs": 100, "max_points": 0})
            raise AssertionError(url)

        def post(self, url: str, json=None):
            raise AssertionError(f"unexpected POST {url}")

    monkeypatch.setattr(mig.httpx, "Client", FakeClient)
    monkeypatch.setattr(mig, "ensure_turbovec_sidecar", lambda *a, **k: True)
    monkeypatch.setattr(mig, "ensure_sparse_sidecar", lambda *a, **k: True)

    code = mig.run_migration(
        qdrant_url="http://127.0.0.1:6333",
        collection="nomad_knowledge_base",
        turbovec_url="http://127.0.0.1:18097",
        sparse_url="http://127.0.0.1:18096",
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "already synced" in out
    assert "Migration complete" in out


def test_run_migration_empty_qdrant_no_sidecars_is_noop(monkeypatch, capsys) -> None:
    from scripts import migrate_qdrant_sidecars as mig

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url: str):
            return FakeResp({"result": {"points_count": 0}})

    monkeypatch.setattr(mig.httpx, "Client", FakeClient)
    code = mig.run_migration(
        qdrant_url="http://127.0.0.1:6333",
        collection="nomad_knowledge_base",
        turbovec_url="",
        sparse_url="",
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Migration complete" in out


def test_run_migration_flushes_sidecars_after_qdrant_clear(monkeypatch, capsys) -> None:
    """After a collection clear the sidecars must be emptied, not left as-is.

    Otherwise BM25 and TurboVec keep returning chunks whose Qdrant points are
    gone, and the proxy injects text the operator believes was deleted.
    """
    from scripts import migrate_qdrant_sidecars as mig

    reindexed: list[str] = []

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url: str):
            if "/collections/" in url:
                return FakeResp({"result": {"points_count": 0}})
            if "18097" in url:
                return FakeResp({"status": "ok", "vectors": 4200})
            return FakeResp({"status": "ok", "docs": 4200, "max_points": 0})

        def post(self, url: str, json=None):
            reindexed.append(url)
            return FakeResp({"docs": 0, "vectors": 0})

    monkeypatch.setattr(mig.httpx, "Client", FakeClient)
    monkeypatch.setattr(mig, "ensure_turbovec_sidecar", lambda *a, **k: True)
    monkeypatch.setattr(mig, "ensure_sparse_sidecar", lambda *a, **k: True)

    code = mig.run_migration(
        qdrant_url="http://127.0.0.1:6333",
        collection="nomad_knowledge_base",
        turbovec_url="http://127.0.0.1:18097",
        sparse_url="http://127.0.0.1:18096",
    )
    assert code == 0
    assert any("18097" in url and "reindex" in url for url in reindexed)
    assert any("18096" in url and "reindex" in url for url in reindexed)
