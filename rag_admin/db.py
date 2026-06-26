"""SQLite persistence for catalog subscriptions in the admin database file.

Ingest state (kb_ingest_state, ingest_jobs) is stored by ingest.db.IngestDatabase
using the same sqlite path (ADMIN_DB_PATH).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ingest.db import IngestDatabase


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AdminDatabase:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.ingest = IngestDatabase(db_path)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS catalog_subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    remote_url TEXT UNIQUE NOT NULL,
                    file_name TEXT,
                    local_path TEXT,
                    remote_size INTEGER,
                    remote_modified TEXT,
                    auto_update INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'subscribed',
                    last_checked TEXT,
                    last_downloaded TEXT,
                    last_error TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                """
            )
            self._ensure_subscription_columns(conn)

    def _ensure_subscription_columns(self, conn: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(catalog_subscriptions)")
        }
        if "package_key" not in columns:
            conn.execute(
                "ALTER TABLE catalog_subscriptions ADD COLUMN package_key TEXT"
            )
        if "catalog_path" not in columns:
            conn.execute(
                "ALTER TABLE catalog_subscriptions ADD COLUMN catalog_path TEXT"
            )

    def get_subscription_by_package(
        self,
        source_id: str,
        package_key: str,
        catalog_path: str,
    ) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM catalog_subscriptions
                WHERE source_id = ? AND package_key = ? AND catalog_path = ?
                """,
                (source_id, package_key, catalog_path),
            ).fetchone()
        return dict(row) if row else None

    def create_subscription(
        self,
        *,
        source_id: str,
        remote_url: str,
        file_name: str,
        local_path: str,
        remote_size: int | None,
        remote_modified: str | None,
        auto_update: bool,
        package_key: str | None = None,
        catalog_path: str | None = None,
    ) -> int:
        now = _utc_now()
        catalog_path = catalog_path or ""
        with self._conn() as conn:
            if package_key:
                existing = conn.execute(
                    """
                    SELECT id FROM catalog_subscriptions
                    WHERE source_id = ? AND package_key = ? AND catalog_path = ?
                    """,
                    (source_id, package_key, catalog_path),
                ).fetchone()
                if existing:
                    conn.execute(
                        """
                        UPDATE catalog_subscriptions SET
                            remote_url = ?,
                            file_name = ?,
                            local_path = ?,
                            remote_size = ?,
                            remote_modified = ?,
                            auto_update = ?,
                            status = 'queued',
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            remote_url,
                            file_name,
                            local_path,
                            remote_size,
                            remote_modified,
                            1 if auto_update else 0,
                            now,
                            existing["id"],
                        ),
                    )
                    return int(existing["id"])

            conn.execute(
                """
                INSERT INTO catalog_subscriptions
                (source_id, remote_url, file_name, local_path, remote_size,
                 remote_modified, auto_update, status, package_key, catalog_path,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'subscribed', ?, ?, ?, ?)
                ON CONFLICT(remote_url) DO UPDATE SET
                    source_id=excluded.source_id,
                    file_name=excluded.file_name,
                    local_path=excluded.local_path,
                    remote_size=excluded.remote_size,
                    remote_modified=excluded.remote_modified,
                    auto_update=excluded.auto_update,
                    package_key=excluded.package_key,
                    catalog_path=excluded.catalog_path,
                    status='queued',
                    updated_at=excluded.updated_at
                """,
                (
                    source_id,
                    remote_url,
                    file_name,
                    local_path,
                    remote_size,
                    remote_modified,
                    1 if auto_update else 0,
                    package_key,
                    catalog_path,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT id FROM catalog_subscriptions WHERE remote_url = ?",
                (remote_url,),
            ).fetchone()
        return int(row["id"])

    def update_subscription(self, sub_id: int, **kwargs: Any) -> None:
        with self._conn() as conn:
            fields = dict(kwargs)
            fields["updated_at"] = _utc_now()
            assignments = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(
                f"UPDATE catalog_subscriptions SET {assignments} WHERE id = ?",
                (*fields.values(), sub_id),
            )

    def list_subscriptions(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM catalog_subscriptions ORDER BY file_name"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_auto_update_subscriptions(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM catalog_subscriptions WHERE auto_update = 1"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_pending_downloads(self, limit: int = 5) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM catalog_subscriptions
                WHERE status IN ('queued', 'update_queued')
                ORDER BY updated_at
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_subscription(self, sub_id: int) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM catalog_subscriptions WHERE id = ?",
                (sub_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_subscription(self, sub_id: int) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM catalog_subscriptions WHERE id = ?",
                (sub_id,),
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM catalog_subscriptions WHERE id = ?", (sub_id,))
        return dict(row)

    def set_subscription_auto_update(self, sub_id: int, enabled: bool) -> None:
        self.update_subscription(sub_id, auto_update=1 if enabled else 0)
