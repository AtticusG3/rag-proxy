"""Background thread: stop nomic embed units when nothing is embedding."""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

from ingest.embed_lifecycle import (
    idle_stop_threshold_sec,
    on_demand_enabled,
    seconds_since_embed_activity,
    stop_idle_embed_units,
)

if TYPE_CHECKING:
    from ingest.worker import IngestWorker
    from rag_admin.job_runner import BackgroundJobRunner

log = logging.getLogger("rag-admin.embed-idle")

JOB_EMBED_POOL_SCALE = "embed_pool_scale"


class EmbedIdleGuard:
    """Poll ingest/proxy embed activity and unload nomic workers when idle."""

    def __init__(
        self,
        worker: IngestWorker,
        job_runner: BackgroundJobRunner,
        *,
        pool_env_path: str,
        poll_sec: float | None = None,
    ) -> None:
        self._worker = worker
        self._job_runner = job_runner
        self._pool_env_path = pool_env_path
        if poll_sec is None:
            poll_sec = float(os.getenv("EMBED_IDLE_POLL_SEC", "15"))
        self._poll_sec = poll_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not on_demand_enabled():
            log.info(
                "embed on-demand idle guard disabled (EMBED_ON_DEMAND=false or no systemctl)"
            )
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="embed-idle-guard",
        )
        self._thread.start()
        log.info("embed idle guard started poll=%ss", self._poll_sec)

    def stop(self) -> None:
        self._stop.set()

    def _ingest_quiet(self) -> bool:
        if self._worker.running_file_count() > 0:
            return False
        pending = self._worker.db.list_pending_files(limit=1)
        return not pending

    def _scale_job_active(self) -> bool:
        return self._job_runner.active_job(JOB_EMBED_POOL_SCALE) is not None

    def _should_stop_embed(self) -> bool:
        if self._scale_job_active():
            return False
        paused = self._worker.paused
        if not paused and not self._ingest_quiet():
            return False
        threshold = idle_stop_threshold_sec(ingest_paused=paused)
        idle_for = seconds_since_embed_activity()
        if idle_for < threshold:
            return False
        return True

    def _run_loop(self) -> None:
        while not self._stop.wait(self._poll_sec):
            try:
                if self._should_stop_embed():
                    stop_idle_embed_units(pool_env_path=self._pool_env_path)
            except Exception:
                log.warning("embed idle guard tick failed", exc_info=True)
