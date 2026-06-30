"""Background subprocess runner for long admin tasks."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_admin.db import AdminDatabase

log = logging.getLogger("rag-admin.jobs")

JOB_MEMGRAPH_BUILD = "memgraph_build"
JOB_EMBED_POOL_SCALE = "embed_pool_scale"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BackgroundJobRunner:
    """Run MemGraphRAG builds, embed pool scaling, and track logs."""

    def __init__(self, db: AdminDatabase, *, repo_root: str, log_dir: str) -> None:
        self.db = db
        self.repo_root = repo_root
        self.log_dir = log_dir
        self._lock = threading.Lock()
        self._running: dict[str, tuple[str, subprocess.Popen[bytes]]] = {}
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    def active_job(self, job_type: str) -> dict[str, Any] | None:
        row = self.db.get_active_background_job(job_type)
        if row is None:
            return None
        return dict(row)

    def _monitor(
        self,
        job_type: str,
        job_id: str,
        proc: subprocess.Popen[bytes],
        log_handle,
        *,
        on_success: Callable[[], None] | None = None,
    ) -> None:
        exit_code = 1
        try:
            exit_code = proc.wait()
        finally:
            log_handle.close()
        status = "done" if exit_code == 0 else "failed"
        message = f"exit code {exit_code}"
        self.db.update_background_job(job_id, status=status, message=message, finished_at=_utc_now())
        with self._lock:
            current = self._running.get(job_type)
            if current is not None and current[0] == job_id:
                del self._running[job_type]
        if on_success is not None and exit_code == 0:
            try:
                on_success()
            except Exception:
                log.exception("background job %s on_success failed", job_id)
        log.info("background job %s (%s) finished: %s", job_id, job_type, message)

    def _start_job(
        self,
        job_type: str,
        cmd: list[str],
        *,
        params: dict[str, Any],
        message: str,
        on_success: Callable[[], None] | None = None,
    ) -> str:
        with self._lock:
            active = self.db.get_active_background_job(job_type)
            if active is not None:
                raise RuntimeError(f"{job_type} already running")
            current = self._running.get(job_type)
            if current is not None and current[1].poll() is None:
                raise RuntimeError(f"{job_type} already running")

            job_id = str(uuid.uuid4())
            log_path = str(Path(self.log_dir) / f"{job_type}_{job_id}.log")
            log_handle = open(log_path, "ab", buffering=0)
            proc = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            self._running[job_type] = (job_id, proc)
            self.db.create_background_job(
                job_id,
                job_type=job_type,
                status="running",
                message=message,
                log_path=log_path,
                pid=proc.pid,
                params_json=json.dumps(params),
            )
            thread = threading.Thread(
                target=self._monitor,
                args=(job_type, job_id, proc, log_handle),
                kwargs={"on_success": on_success},
                daemon=True,
                name=f"{job_type}-{job_id[:8]}",
            )
            thread.start()
            return job_id

    def start_memgraph_build(self, params: dict[str, Any]) -> str:
        python = sys.executable
        script = os.path.join(self.repo_root, "scripts", "build_memgraphrag_index.py")
        cmd = [
            python,
            script,
            "--source",
            "qdrant",
            "--qdrant-url",
            str(params["qdrant_url"]),
            "--collection",
            str(params["collection"]),
            "--output",
            str(params["output"]),
            "--llm-url",
            str(params["llm_url"]),
            "--llm-model",
            str(params["llm_model"]),
            "--max-chunks",
            str(params["max_chunks"]),
            "--concurrency",
            str(params["concurrency"]),
            "--embed-url",
            str(params["embed_url"]),
        ]
        if params.get("skip_relations"):
            cmd.append("--skip-relations")
        return self._start_job(
            JOB_MEMGRAPH_BUILD,
            cmd,
            params=params,
            message="MemGraphRAG index build started",
        )

    def start_embed_pool_scale(
        self,
        params: dict[str, Any],
        *,
        on_success: Callable[[], None] | None = None,
    ) -> str:
        python = sys.executable
        script = os.path.join(self.repo_root, "scripts", "scale_nomic_embed_pool.py")
        cmd = [
            python,
            script,
            "--apply",
            "--pool-env",
            str(params["pool_env_path"]),
            "--scale-env",
            str(params["scale_env_path"]),
        ]
        return self._start_job(
            JOB_EMBED_POOL_SCALE,
            cmd,
            params=params,
            message="Embed pool scale started",
            on_success=on_success,
        )

    def stop_active(self, job_type: str) -> bool:
        with self._lock:
            entry = self._running.get(job_type)
        if entry is None:
            return False
        job_id, proc = entry
        if proc.poll() is not None:
            return False
        proc.terminate()
        self.db.update_background_job(
            job_id,
            status="failed",
            message="stopped by operator",
            finished_at=_utc_now(),
        )
        return True

    def tail_log(self, job_id: str, *, max_bytes: int = 8000) -> str:
        row = self.db.get_background_job(job_id)
        if row is None:
            return ""
        log_path = row.get("log_path")
        if not log_path or not os.path.isfile(log_path):
            return ""
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            data = handle.read()
        return data.decode("utf-8", errors="replace")
