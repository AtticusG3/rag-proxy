"""Background ingest worker."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from ingest.chunking import chunk_text
from ingest.embedder import embed_texts
from ingest.pdf_reader import read_pdf_text
from ingest.qdrant_writer import build_point, delete_by_source, ensure_collection, upsert_points
from ingest.scanner import scan_storage
from ingest.types import determine_file_type
from ingest.zim_reader import iter_zim_articles

log = logging.getLogger("ingest.worker")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IngestConfig:
    zim_dir: str
    upload_dir: str
    embed_url: str
    qdrant_url: str
    qdrant_collection: str
    sparse_index_url: str
    batch_size: int = 32
    max_articles: int = 0
    embed_max_chars: int = 2000


UpdateStateFn = Callable[..., None]


def _read_text_file(path: str) -> tuple[str, str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    title = Path(path).stem.replace("_", " ").title()
    return title, text


def _iter_chunks_for_file(
    file_path: str,
    *,
    max_articles: int,
) -> list[tuple[str, str, str]]:
    """Return list of (title, url_or_path, chunk_text) tuples."""
    file_type = determine_file_type(file_path)
    source = file_path
    out: list[tuple[str, str, str]] = []

    if file_type == "text":
        title, text = _read_text_file(file_path)
        for idx, piece in enumerate(chunk_text(text)):
            out.append((title, source, piece))
        return out

    if file_type == "zim":
        for article in iter_zim_articles(file_path, max_articles=max_articles):
            for idx, piece in enumerate(chunk_text(article.text)):
                out.append((article.title, source, piece))
        return out

    if file_type == "pdf":
        title, text = read_pdf_text(file_path)
        if not text.strip():
            return out
        for idx, piece in enumerate(chunk_text(text)):
            out.append((title, source, piece))
        return out

    raise ValueError(f"Unsupported file type for {file_path}")


def process_file(
    file_path: str,
    config: IngestConfig,
    *,
    on_progress: UpdateStateFn | None = None,
) -> int:
    """Embed one file into Qdrant. Returns total chunks embedded."""
    ensure_collection(config.qdrant_url, config.qdrant_collection)
    chunks = _iter_chunks_for_file(file_path, max_articles=config.max_articles)
    if not chunks:
        return 0

    total = 0
    batch_size = max(1, config.batch_size)
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c[2] for c in batch]
        embeddings = embed_texts(
            texts,
            embed_url=config.embed_url,
            max_chars=config.embed_max_chars,
        )
        points = []
        for i, (title, source, text) in enumerate(batch):
            chunk_idx = batch_start + i
            points.append(
                build_point(
                    text=text,
                    source=source,
                    title=title,
                    chunk_idx=chunk_idx,
                    embedding=embeddings[i],
                )
            )
        upsert_points(config.qdrant_url, config.qdrant_collection, points)
        total += len(points)
        if on_progress:
            on_progress(chunks_embedded=total)

    return total


def trigger_sparse_reindex(config: IngestConfig) -> int | None:
    if not config.sparse_index_url:
        return None
    url = f"{config.sparse_index_url.rstrip('/')}/reindex"
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(url, json={"collection": config.qdrant_collection})
            response.raise_for_status()
            return int(response.json().get("docs", 0))
    except Exception as exc:
        log.warning("sparse reindex failed: %s", exc)
        return None


class IngestWorker:
    """Single-threaded job processor with SQLite-backed state."""

    def __init__(
        self,
        config: IngestConfig,
        db_module: object,
    ) -> None:
        self.config = config
        self.db = db_module
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ingest-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def enqueue_sync(self) -> str:
        """Scan storage and queue only new or previously failed files."""
        job_id = str(uuid.uuid4())
        self.db.create_job(job_id, job_type="sync")
        directories = [self.config.zim_dir, self.config.upload_dir]
        queued_new = 0
        queued_retry = 0
        for path in scan_storage(*directories):
            row = self.db.get_file_state(path)
            if row is None:
                self.db.upsert_file_state(
                    path,
                    status="pending",
                    file_type=determine_file_type(path),
                )
                queued_new += 1
            elif row["status"] == "failed":
                self.db.retry_file_state(path)
                queued_retry += 1
        message = f"scan: {queued_new} new, {queued_retry} retries (indexed files unchanged)"
        self.db.update_job(job_id, status="queued", message=message)
        return job_id

    def retry_file(self, file_path: str) -> str:
        job_id = str(uuid.uuid4())
        self.db.create_job(job_id, job_type="retry", message=file_path)
        if not self.db.retry_file_state(file_path):
            self.db.upsert_file_state(
                file_path,
                status="pending",
                file_type=determine_file_type(file_path),
            )
        return job_id

    def retry_all_failed(self) -> str:
        job_id = str(uuid.uuid4())
        failed = self.db.list_failed_files()
        self.db.create_job(job_id, job_type="retry_failed")
        for row in failed:
            self.db.retry_file_state(row["file_path"])
        self.db.update_job(
            job_id,
            status="queued",
            message=f"{len(failed)} failed file(s) re-queued",
        )
        return job_id

    def enqueue_file(self, file_path: str) -> str:
        job_id = str(uuid.uuid4())
        self.db.create_job(job_id, job_type="file", message=file_path)
        self.db.upsert_file_state(file_path, status="pending")
        return job_id

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            pending = self.db.list_pending_files(limit=1)
            if not pending:
                time.sleep(1.0)
                continue
            file_path = pending[0]["file_path"]
            try:
                self._process_one(file_path)
            except Exception as exc:
                log.exception("ingest failed for %s", file_path)
                self.db.update_file_state(
                    file_path,
                    status="failed",
                    last_error=str(exc),
                    finished_at=_utc_now(),
                )

    def _process_one(self, file_path: str) -> None:
        with self._lock:
            self.db.update_file_state(
                file_path,
                status="running",
                started_at=_utc_now(),
                last_error=None,
            )

            def on_progress(**kwargs: object) -> None:
                chunks = kwargs.get("chunks_embedded")
                if isinstance(chunks, int):
                    self.db.update_file_state(file_path, chunks_embedded=chunks)

            count = process_file(file_path, self.config, on_progress=on_progress)
            self.db.update_file_state(
                file_path,
                status="indexed",
                chunks_embedded=count,
                finished_at=_utc_now(),
            )
            trigger_sparse_reindex(self.config)

    def remove_file_from_index(self, file_path: str) -> None:
        delete_by_source(
            self.config.qdrant_url,
            self.config.qdrant_collection,
            file_path,
        )
        if os.path.isfile(file_path):
            os.remove(file_path)
        self.db.delete_file_state(file_path)
        trigger_sparse_reindex(self.config)
