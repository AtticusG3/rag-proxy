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
                CREATE TABLE IF NOT EXISTS admin_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS background_jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    log_path TEXT,
                    pid INTEGER,
                    params_json TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS admin_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT,
                    client_ip TEXT
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

    def get_setting(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM admin_settings WHERE key = ?",
                (key,),
            ).fetchone()
        return str(row["value"]) if row else None

    def set_setting(self, key: str, value: str) -> None:
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO admin_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, now),
            )

    def delete_setting(self, key: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM admin_settings WHERE key = ?", (key,))

    def create_background_job(
        self,
        job_id: str,
        *,
        job_type: str,
        status: str,
        message: str,
        log_path: str,
        pid: int,
        params_json: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO background_jobs
                (id, job_type, status, message, log_path, pid, params_json, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, job_type, status, message, log_path, pid, params_json, _utc_now()),
            )

    def update_background_job(self, job_id: str, **fields: object) -> None:
        allowed = {"status", "message", "finished_at"}
        assignments: list[str] = []
        values: list[object] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            assignments.append(f"{key} = ?")
            values.append(value)
        if not assignments:
            return
        values.append(job_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE background_jobs SET {', '.join(assignments)} WHERE id = ?",
                values,
            )

    def get_background_job(self, job_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM background_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_active_background_job(self, job_type: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM background_jobs
                WHERE job_type = ? AND status = 'running'
                ORDER BY started_at DESC LIMIT 1
                """,
                (job_type,),
            ).fetchone()
        return dict(row) if row else None

    def list_background_jobs(self, job_type: str, *, limit: int = 10) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM background_jobs
                WHERE job_type = ?
                ORDER BY started_at DESC LIMIT ?
                """,
                (job_type, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_admin_session(
        self,
        session_id: str,
        *,
        expires_at: str,
        client_ip: str | None = None,
    ) -> None:
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO admin_sessions
                (session_id, created_at, expires_at, revoked_at, client_ip)
                VALUES (?, ?, ?, NULL, ?)
                """,
                (session_id, now, expires_at, client_ip),
            )

    def get_admin_session(self, session_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM admin_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def revoke_admin_session(self, session_id: str) -> None:
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE admin_sessions
                SET revoked_at = ?
                WHERE session_id = ? AND revoked_at IS NULL
                """,
                (now, session_id),
            )

    def prune_expired_admin_sessions(self) -> int:
        """Delete sessions past expires_at (call on login to limit table growth)."""
        now = _utc_now()
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM admin_sessions WHERE expires_at < ?",
                (now,),
            )
        return int(cursor.rowcount)
