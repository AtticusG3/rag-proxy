"""Tests for embed pool scale background jobs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_admin.db import AdminDatabase
from rag_admin.job_runner import JOB_EMBED_POOL_SCALE, BackgroundJobRunner
from rag_admin.settings_store import SettingsStore


def test_sync_pool_ingest_from_pool_env_writes_admin_env(tmp_path: Path) -> None:
    pool_env = tmp_path / "pool.env"
    admin_env = tmp_path / "admin.env"
    pool_env.write_text(
        "INGEST_EMBED_URLS=http://127.0.0.1:18089,http://127.0.0.1:18090\n"
        "INGEST_EMBED_CONCURRENCY=32\n",
        encoding="utf-8",
    )
    store = SettingsStore(
        AdminDatabase(str(tmp_path / "admin.sqlite")),
        admin_env_path=str(admin_env),
        proxy_env_path=str(tmp_path / "proxy.env"),
        pool_scale_env_path=str(tmp_path / "scale.env"),
        pool_env_path=str(pool_env),
    )
    synced = store.sync_pool_ingest_from_pool_env()
    assert sorted(synced) == ["INGEST_EMBED_CONCURRENCY", "INGEST_EMBED_URLS"]
    assert "INGEST_EMBED_URLS" in admin_env.read_text(encoding="utf-8")
    config = store.build_ingest_config(zim_dir=str(tmp_path), upload_dir=str(tmp_path))
    assert config.embed_concurrency == 32
    assert len(config.embed_urls or []) == 2


def test_sync_pool_ingest_syncs_capacity_plan_keys(tmp_path: Path) -> None:
    pool_env = tmp_path / "pool.env"
    admin_env = tmp_path / "admin.env"
    pool_env.write_text(
        "INGEST_EMBED_URLS=http://127.0.0.1:18089\n"
        "INGEST_EMBED_CONCURRENCY=12\n"
        "INGEST_FILE_CONCURRENCY=3\n"
        "INGEST_BATCH_SIZE=32\n"
        "INGEST_CHUNK_CONCURRENCY=2\n"
        "INGEST_CHUNK_SEMANTIC=false\n"
        "INGEST_SPARSE_REINDEX=off\n"
        "CAPACITY_CPU_CORES=16\n",
        encoding="utf-8",
    )
    store = SettingsStore(
        AdminDatabase(str(tmp_path / "admin.sqlite")),
        admin_env_path=str(admin_env),
        proxy_env_path=str(tmp_path / "proxy.env"),
        pool_scale_env_path=str(tmp_path / "scale.env"),
        pool_env_path=str(pool_env),
    )
    synced = store.sync_pool_ingest_from_pool_env()
    # All planner-owned ingest knobs sync; host snapshot keys stay in the pool env.
    assert "INGEST_FILE_CONCURRENCY" in synced
    assert "INGEST_CHUNK_CONCURRENCY" in synced
    assert "INGEST_CHUNK_SEMANTIC" in synced
    assert "CAPACITY_CPU_CORES" not in synced
    config = store.build_ingest_config(zim_dir=str(tmp_path), upload_dir=str(tmp_path))
    assert config.file_concurrency == 3
    assert config.chunk_concurrency == 2
    assert config.batch_size == 32
    assert config.chunk_config.semantic_enabled is False
    assert config.sparse_reindex_mode == "off"


@patch("rag_admin.job_runner.threading.Thread")
@patch("rag_admin.job_runner.subprocess.Popen")
def test_start_embed_pool_scale_registers_job(
    mock_popen: MagicMock, mock_thread: MagicMock, tmp_path: Path
) -> None:
    mock_proc = MagicMock()
    mock_proc.pid = 4242
    mock_proc.poll.return_value = None
    mock_proc.wait.return_value = 0
    mock_popen.return_value = mock_proc
    mock_thread.return_value.start = MagicMock()

    db = AdminDatabase(str(tmp_path / "admin.sqlite"))
    runner = BackgroundJobRunner(
        db,
        repo_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
    )
    job_id = runner.start_embed_pool_scale(
        {
            "pool_env_path": "/opt/ai/config/nomic-embed-pool.env",
            "scale_env_path": "/opt/ai/config/nomic-embed-scale.env",
            "semantic_requested": "true",
        }
    )
    assert job_id
    active = runner.active_job(JOB_EMBED_POOL_SCALE)
    assert active is not None
    assert active["job_type"] == JOB_EMBED_POOL_SCALE
    cmd = mock_popen.call_args[0][0]
    assert any("run_ingest_capacity_scale.py" in part for part in cmd)
    assert "--semantic-requested" in cmd


def test_active_job_clears_orphaned_running_row_so_scale_can_start(tmp_path: Path) -> None:
    """Stale status=running after restart must not permanently disable Scale."""
    db = AdminDatabase(str(tmp_path / "admin.sqlite"))
    db.create_background_job(
        "dead-job-id",
        job_type=JOB_EMBED_POOL_SCALE,
        status="running",
        message="left behind after restart",
        log_path=str(tmp_path / "gone.log"),
        pid=999_999_999,  # not a live process
        params_json="{}",
    )
    runner = BackgroundJobRunner(
        db,
        repo_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
    )
    assert runner.active_job(JOB_EMBED_POOL_SCALE) is None
    row = db.get_background_job("dead-job-id")
    assert row is not None
    assert row["status"] == "failed"
    assert "orphaned" in (row["message"] or "")


def test_reconcile_orphans_clears_dead_jobs_on_startup_entry_point(tmp_path: Path) -> None:
    """Lifespan calls reconcile_orphans() so Scale/Start work after admin restart."""
    from rag_admin.job_runner import JOB_MEMGRAPH_BUILD

    db = AdminDatabase(str(tmp_path / "admin.sqlite"))
    db.create_background_job(
        "orphan-scale",
        job_type=JOB_EMBED_POOL_SCALE,
        status="running",
        message="left behind",
        log_path=str(tmp_path / "scale.log"),
        pid=999_999_997,
        params_json="{}",
    )
    db.create_background_job(
        "orphan-memgraph",
        job_type=JOB_MEMGRAPH_BUILD,
        status="running",
        message="left behind",
        log_path=str(tmp_path / "mem.log"),
        pid=999_999_996,
        params_json="{}",
    )
    runner = BackgroundJobRunner(
        db,
        repo_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
    )
    assert runner.reconcile_orphans() == 2
    assert runner.reconcile_orphans() == 0
    for job_id in ("orphan-scale", "orphan-memgraph"):
        row = db.get_background_job(job_id)
        assert row is not None
        assert row["status"] == "failed"
        assert "orphaned" in (row["message"] or "")


def test_reconcile_orphans_keeps_row_when_pid_still_alive(tmp_path: Path) -> None:
    """Must not mark a live job failed just because _running is empty after restart."""
    db = AdminDatabase(str(tmp_path / "admin.sqlite"))
    db.create_background_job(
        "live-job",
        job_type=JOB_EMBED_POOL_SCALE,
        status="running",
        message="still running",
        log_path=str(tmp_path / "live.log"),
        pid=42,
        params_json="{}",
    )
    runner = BackgroundJobRunner(
        db,
        repo_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
    )
    with patch("rag_admin.job_runner._pid_alive", return_value=True):
        assert runner.reconcile_orphans() == 0
    row = db.get_background_job("live-job")
    assert row is not None
    assert row["status"] == "running"


def test_stop_active_clears_orphaned_db_row_without_popen_handle(tmp_path: Path) -> None:
    """Stop scale must work after restart when only the SQLite row remains."""
    db = AdminDatabase(str(tmp_path / "admin.sqlite"))
    db.create_background_job(
        "orphan-stop",
        job_type=JOB_EMBED_POOL_SCALE,
        status="running",
        message="orphaned",
        log_path=str(tmp_path / "gone.log"),
        pid=999_999_998,
        params_json="{}",
    )
    runner = BackgroundJobRunner(
        db,
        repo_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
    )
    # Bypass active_job reconcile so stop sees the stale row.
    assert db.get_active_background_job(JOB_EMBED_POOL_SCALE) is not None
    assert runner.stop_active(JOB_EMBED_POOL_SCALE) is True
    row = db.get_background_job("orphan-stop")
    assert row is not None
    assert row["status"] == "failed"
    assert row["message"] == "stopped by operator"
