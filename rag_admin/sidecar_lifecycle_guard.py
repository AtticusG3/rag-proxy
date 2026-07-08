"""Background thread: stop BM25 sidecar during ingest, restart when idle."""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

from ingest.sidecar_lifecycle import (
    ensure_sparse_sidecar,
    sidecar_on_demand_enabled,
    stop_sparse_sidecar,
)

if TYPE_CHECKING:
    from ingest.worker import IngestWorker

log = logging.getLogger("rag-admin.sidecar-lifecycle")


class SidecarLifecycleGuard:
    """Free RAM during bulk ingest by stopping the sparse sidecar systemd unit."""

    def __init__(
        self,
        worker: IngestWorker,
        *,
        sparse_index_url: str,
        poll_sec: float | None = None,
    ) -> None:
        self._worker = worker
        self._sparse_index_url = sparse_index_url.strip()
        if poll_sec is None:
            poll_sec = float(os.getenv("SIDECAR_LIFECYCLE_POLL_SEC", "20"))
        self._poll_sec = poll_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ingest_was_active = False

    def start(self) -> None:
        if not sidecar_on_demand_enabled():
            log.info("sidecar lifecycle guard disabled (SIDECAR_ON_DEMAND=false or no systemctl)")
            return
        if not self._sparse_index_url:
            log.info("sidecar lifecycle guard disabled (no SPARSE_INDEX_URL)")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="sidecar-lifecycle-guard",
        )
        self._thread.start()
        log.info("sidecar lifecycle guard started poll=%ss", self._poll_sec)

    def stop(self) -> None:
        self._stop.set()

    def _ingest_active(self) -> bool:
        if self._worker.running_file_count() > 0:
            return True
        return bool(self._worker.db.list_pending_files(limit=1))

    def _run_loop(self) -> None:
        while not self._stop.wait(self._poll_sec):
            try:
                active = self._ingest_active()
                if active:
                    if not self._ingest_was_active:
                        stop_sparse_sidecar()
                    self._ingest_was_active = True
                    continue
                if self._ingest_was_active:
                    log.info("ingest idle: ensuring sparse sidecar is up before reindex")
                    ensure_sparse_sidecar(self._sparse_index_url, wait_health=True)
                self._ingest_was_active = False
            except Exception:
                log.warning("sidecar lifecycle guard tick failed", exc_info=True)
