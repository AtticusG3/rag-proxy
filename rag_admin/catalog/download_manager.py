"""Download queue for subscribed catalog packages."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import httpx

from ingest.types import determine_file_type
from rag_admin.catalog.providers import browse_source, fetch_remote_meta
from rag_admin.catalog.zim_versions import KIWIX_SOURCES, parse_zim_stamp, resolve_latest_item

if TYPE_CHECKING:
    from ingest.worker import IngestWorker

log = logging.getLogger("catalog.download")
USER_AGENT = "rag-admin/1.0 (local-ai-infra)"


class CatalogDownloadManager:
    """Background downloader for catalog subscriptions."""

    def __init__(
        self,
        db: Any,
        zim_dir: str,
        upload_dir: str,
        ingest_worker: IngestWorker | None = None,
    ) -> None:
        self.db = db
        self.zim_dir = zim_dir
        self.upload_dir = upload_dir
        self.ingest_worker = ingest_worker
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        os.makedirs(zim_dir, exist_ok=True)
        os.makedirs(upload_dir, exist_ok=True)

    def _local_path_for(self, source_id: str, remote_url: str) -> str:
        file_name = unquote(Path(remote_url.rstrip("/")).name.split("?")[0])
        if source_id == "archive" and "/download/" in remote_url:
            tail = remote_url.split("/download/", 1)[1]
            parts = tail.split("/", 1)
            if len(parts) == 2:
                file_name = f"{parts[0]}_{parts[1]}"
        if source_id == "arxiv" and file_name.endswith(".pdf"):
            file_name = file_name.replace("/", "_")
        ext = Path(file_name).suffix.lower()
        base_dir = self.zim_dir if ext == ".zim" else self.upload_dir
        return os.path.join(base_dir, file_name)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="catalog-download"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def subscribe(
        self,
        source_id: str,
        remote_url: str,
        *,
        auto_update: bool = True,
        package_key: str | None = None,
        catalog_path: str | None = None,
    ) -> int:
        meta = fetch_remote_meta(remote_url)
        dest_path = self._local_path_for(source_id, remote_url)
        file_name = Path(dest_path).name
        if not package_key:
            parsed = parse_zim_stamp(file_name)
            if parsed is not None:
                package_key = parsed[0]
        # Prefer the currently indexed path when bumping a dated package so the
        # downloader can retire the previous archive after the new file lands.
        local_path = dest_path
        if package_key:
            existing = self.db.get_subscription_by_package(
                source_id, package_key, catalog_path or ""
            )
            prev = (existing or {}).get("local_path")
            if prev and str(prev) != dest_path:
                local_path = str(prev)
        sub_id = self.db.create_subscription(
            source_id=source_id,
            remote_url=remote_url,
            file_name=file_name,
            local_path=local_path,
            remote_size=meta.get("size_bytes"),
            remote_modified=meta.get("modified"),
            auto_update=auto_update,
            package_key=package_key,
            catalog_path=catalog_path or "",
        )
        self.db.update_subscription(sub_id, status="queued")
        return sub_id

    def _retire_local_file(self, path: str) -> None:
        """Drop a local archive from indexes and disk (no-op if path empty)."""
        if not path:
            return
        if self.ingest_worker is not None:
            self.ingest_worker.remove_file_from_index(path)
            return
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                log.warning("failed to remove superseded file %s", path)

    def _purge_superseded_package_files(
        self, *, package_key: str | None, keep_path: str
    ) -> None:
        """Remove older dated ZIMs for the same package; keep only keep_path.

        Scans zim_dir and ingest file_state so orphans left by prior version bumps
        (when local_path was rewritten before the old file was retired) are healed.
        """
        if not package_key or not keep_path:
            return
        keep_name = Path(keep_path).name.lower()
        candidates: set[str] = set()
        if os.path.isdir(self.zim_dir):
            for name in os.listdir(self.zim_dir):
                if name.endswith(".part"):
                    continue
                parsed = parse_zim_stamp(name)
                if parsed is None or parsed[0] != package_key:
                    continue
                if name.lower() == keep_name:
                    continue
                candidates.add(os.path.join(self.zim_dir, name))
        ingest = getattr(self.db, "ingest", None)
        if ingest is not None:
            for row in ingest.list_file_states():
                name = row.get("file_name") or Path(row["file_path"]).name
                parsed = parse_zim_stamp(str(name))
                if parsed is None or parsed[0] != package_key:
                    continue
                if str(name).lower() == keep_name:
                    continue
                candidates.add(str(row["file_path"]))
        for path in sorted(candidates):
            if Path(path).name.lower() == keep_name:
                continue
            log.info(
                "retiring superseded package file %s (keep %s)",
                path,
                keep_path,
            )
            self._retire_local_file(path)

    def _maybe_queue_newer_zim(self, row: dict[str, Any]) -> bool:
        package_key = row.get("package_key")
        catalog_path = row.get("catalog_path")
        if row.get("source_id") not in KIWIX_SOURCES or not package_key:
            return False
        listing = browse_source(row["source_id"], catalog_path or "")
        latest = resolve_latest_item(listing.get("items", []), package_key)
        if latest is None or latest.url == row.get("remote_url"):
            return False
        # Keep local_path on the currently indexed file until download finishes so
        # _download_one can retire it. Only remote_url/file_name point at the newer ZIM.
        self.db.update_subscription(
            row["id"],
            remote_url=latest.url,
            file_name=latest.name,
            status="update_queued",
            last_error=None,
        )
        return True

    def check_updates(self) -> list[int]:
        """Queue re-downloads for subscriptions with newer remote copies."""
        queued: list[int] = []
        for row in self.db.list_auto_update_subscriptions():
            try:
                if self._maybe_queue_newer_zim(row):
                    queued.append(row["id"])
                    self.db.update_subscription(
                        row["id"],
                        last_checked=time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    )
                    continue
                meta = fetch_remote_meta(row["remote_url"])
            except Exception as exc:
                self.db.update_subscription(
                    row["id"],
                    last_checked=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    last_error=str(exc),
                )
                continue
            local_size = None
            if os.path.isfile(row["local_path"]):
                local_size = os.path.getsize(row["local_path"])
            remote_size = meta.get("size_bytes")
            needs = False
            if remote_size and local_size and remote_size != local_size:
                needs = True
            if not os.path.isfile(row["local_path"]):
                needs = True
            remote_mod = meta.get("modified")
            if remote_mod and row.get("remote_modified") and remote_mod != row["remote_modified"]:
                needs = True
            self.db.update_subscription(
                row["id"],
                last_checked=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                remote_size=remote_size,
                remote_modified=remote_mod,
            )
            if needs:
                self.db.update_subscription(row["id"], status="update_queued")
                queued.append(row["id"])
            else:
                # Already on latest remote URL: still drop older dated siblings left
                # on disk / in ingest from earlier buggy version bumps.
                keep = self._local_path_for(row["source_id"], row["remote_url"])
                self._purge_superseded_package_files(
                    package_key=row.get("package_key"),
                    keep_path=keep,
                )
                if keep != row.get("local_path"):
                    self.db.update_subscription(
                        row["id"],
                        local_path=keep,
                        file_name=Path(keep).name,
                    )
        return queued

    def process_pending_downloads(self, max_items: int = 10) -> int:
        """Process download queue synchronously (for cron/CLI)."""
        done = 0
        while done < max_items:
            pending = self.db.list_pending_downloads(limit=1)
            if not pending:
                break
            self._download_one(pending[0])
            done += 1
        return done

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            pending = self.db.list_pending_downloads(limit=1)
            if not pending:
                time.sleep(2.0)
                continue
            row = pending[0]
            try:
                self._download_one(row)
            except Exception as exc:
                log.exception("download failed for %s", row.get("remote_url"))
                self.db.update_subscription(
                    row["id"],
                    status="failed",
                    last_error=str(exc),
                )

    def _download_one(self, row: dict[str, Any]) -> None:
        sub_id = row["id"]
        remote_url = row["remote_url"]
        # Destination always follows remote_url so dated version bumps land on the
        # new filename even when local_path still points at the previous archive.
        local_path = self._local_path_for(row["source_id"], remote_url)
        previous_path = row.get("local_path") or ""
        self.db.update_subscription(sub_id, status="downloading", last_error=None)

        same_path = bool(previous_path) and os.path.normpath(
            previous_path
        ) == os.path.normpath(local_path)
        if same_path and os.path.isfile(local_path) and self.ingest_worker:
            from ingest.qdrant_writer import delete_by_source

            delete_by_source(
                self.ingest_worker.config.qdrant_url,
                self.ingest_worker.config.qdrant_collection,
                local_path,
            )

        tmp_path = f"{local_path}.part"
        meta = fetch_remote_meta(remote_url)
        with httpx.Client(
            timeout=600.0,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            with client.stream("GET", remote_url) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        if self._stop.is_set():
                            raise InterruptedError("catalog download stopped")
                        if chunk:
                            handle.write(chunk)
        if os.path.isfile(tmp_path) and self._stop.is_set():
            os.remove(tmp_path)
            raise InterruptedError("catalog download stopped")
        os.replace(tmp_path, local_path)

        self.db.update_subscription(
            sub_id,
            status="downloaded",
            file_name=Path(local_path).name,
            local_path=local_path,
            remote_size=meta.get("size_bytes"),
            remote_modified=meta.get("modified"),
            last_downloaded=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_error=None,
        )
        self.db.ingest.upsert_file_state(
            local_path,
            status="pending",
            file_type=determine_file_type(local_path),
        )
        if self.ingest_worker:
            self.ingest_worker.enqueue_file(local_path)

        package_key = row.get("package_key")
        if not package_key:
            parsed = parse_zim_stamp(Path(local_path).name)
            if parsed is not None:
                package_key = parsed[0]
        self._purge_superseded_package_files(
            package_key=package_key,
            keep_path=local_path,
        )

    def retry_subscription(self, sub_id: int) -> bool:
        row = self.db.get_subscription(sub_id)
        if row is None:
            return False
        if row.get("status") not in ("failed",):
            return False
        self.db.update_subscription(sub_id, status="queued", last_error=None)
        return True
