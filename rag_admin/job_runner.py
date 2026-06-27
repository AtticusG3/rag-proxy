"""Background subprocess runner for long admin tasks."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_admin.db import AdminDatabase

log = logging.getLogger("rag-admin.jobs")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BackgroundJobRunner:
    """Run MemGraphRAG index builds and track stdout/stderr in log files."""

    def __init__(self, db: AdminDatabase, *, repo_root: str, log_dir: str) -> None:
        self.db = db
        self.repo_root = repo_root
        self.log_dir = log_dir
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._job_id: str | None = None
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    def active_job(self) -> dict[str, Any] | None:
        row = self.db.get_active_background_job("memgraph_build")
        if row is None:
            return None
        return dict(row)

    def _monitor(self, job_id: str, proc: subprocess.Popen[bytes], log_handle) -> None:
        exit_code = 1
        try:
            exit_code = proc.wait()
        finally:
            log_handle.close()
        status = "done" if exit_code == 0 else "failed"
        message = f"exit code {exit_code}"
        self.db.update_background_job(job_id, status=status, message=message, finished_at=_utc_now())
        with self._lock:
            if self._job_id == job_id:
                self._proc = None
                self._job_id = None
        log.info("background job %s finished: %s", job_id, message)

    def start_memgraph_build(self, params: dict[str, Any]) -> str:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("MemGraphRAG build already running")
            active = self.db.get_active_background_job("memgraph_build")
            if active is not None:
                raise RuntimeError("MemGraphRAG build already running")

            job_id = str(uuid.uuid4())
            log_path = str(Path(self.log_dir) / f"memgraph_build_{job_id}.log")
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

            log_handle = open(log_path, "ab", buffering=0)
            proc = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            self._proc = proc
            self._job_id = job_id
            self.db.create_background_job(
                job_id,
                job_type="memgraph_build",
                status="running",
                message="MemGraphRAG index build started",
                log_path=log_path,
                pid=proc.pid,
                params_json=json.dumps(params),
            )
            thread = threading.Thread(
                target=self._monitor,
                args=(job_id, proc, log_handle),
                daemon=True,
                name=f"memgraph-build-{job_id[:8]}",
            )
            thread.start()
            return job_id

    def stop_active(self) -> bool:
        with self._lock:
            proc = self._proc
            job_id = self._job_id
        if proc is None or proc.poll() is not None:
            return False
        proc.terminate()
        if job_id:
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
