"""SQLite persistence for ingest state and jobs."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


VALID_PRIORITIES = ("high", "mid", "low")
DEFAULT_PRIORITY = "mid"

# Lower rank = claimed first. Keep in sync with rag_admin.ingest_status._PRIORITY_RANK.
_PRIORITY_ORDER_SQL = (
    "CASE priority WHEN 'high' THEN 0 WHEN 'mid' THEN 1 WHEN 'low' THEN 2 ELSE 1 END"
)


def _sqlite_supports_returning() -> bool:
    """RETURNING on UPDATE requires SQLite 3.35+."""
    parts = sqlite3.sqlite_version.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    return (major, minor) >= (3, 35)


class IngestDatabase:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS kb_ingest_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_name TEXT,
                    file_type TEXT,
                    status TEXT DEFAULT 'pending',
                    priority TEXT DEFAULT 'mid',
                    chunks_embedded INTEGER DEFAULT 0,
                    last_error TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT DEFAULT 'queued',
                    message TEXT,
                    created_at TEXT,
                    finished_at TEXT
                );
                """
            )
            self._ensure_ingest_columns(conn)

    def _ensure_ingest_columns(self, conn: sqlite3.Connection) -> None:
        """Additive migration for kb_ingest_state columns added after v1."""
        columns = {row[1] for row in conn.execute("PRAGMA table_info(kb_ingest_state)")}
        if "priority" not in columns:
            conn.execute(
                "ALTER TABLE kb_ingest_state ADD COLUMN priority TEXT DEFAULT 'mid'"
            )

    def upsert_file_state(
        self,
        file_path: str,
        *,
        status: str | None = None,
        file_type: str | None = None,
        chunks_embedded: int | None = None,
        last_error: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        name = Path(file_path).name
        now = _utc_now()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM kb_ingest_state WHERE file_path = ?",
                (file_path,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO kb_ingest_state
                    (file_path, file_name, file_type, status, chunks_embedded,
                     last_error, started_at, finished_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_path,
                        name,
                        file_type or "",
                        status or "pending",
                        chunks_embedded or 0,
                        last_error,
                        started_at,
                        finished_at,
                        now,
                    ),
                )
            else:
                fields: dict[str, Any] = {"updated_at": now}
                if status is not None:
                    fields["status"] = status
                if file_type is not None:
                    fields["file_type"] = file_type
                if chunks_embedded is not None:
                    fields["chunks_embedded"] = chunks_embedded
                if last_error is not None:
                    fields["last_error"] = last_error
                if started_at is not None:
                    fields["started_at"] = started_at
                if finished_at is not None:
                    fields["finished_at"] = finished_at
                assignments = ", ".join(f"{k} = ?" for k in fields)
                conn.execute(
                    f"UPDATE kb_ingest_state SET {assignments} WHERE file_path = ?",
                    (*fields.values(), file_path),
                )

    def update_file_state(self, file_path: str, **kwargs: Any) -> None:
        self.upsert_file_state(file_path, **kwargs)

    def delete_file_state(self, file_path: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM kb_ingest_state WHERE file_path = ?", (file_path,))

    def get_file_state(self, file_path: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM kb_ingest_state WHERE file_path = ?",
                (file_path,),
            ).fetchone()
        return dict(row) if row else None

    def set_file_priority(self, file_path: str, priority: str) -> bool:
        """Set claim priority for one file. Returns False if the file is unknown."""
        if priority not in VALID_PRIORITIES:
            raise ValueError(f"invalid priority: {priority}")
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE kb_ingest_state SET priority = ? WHERE file_path = ?",
                (priority, file_path),
            )
            return cursor.rowcount > 0

    def retry_file_state(self, file_path: str, *, reset_chunks: bool = False) -> bool:
        """Re-queue one file for ingest without touching indexed rows."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM kb_ingest_state WHERE file_path = ?",
                (file_path,),
            ).fetchone()
            if row is None:
                return False
            chunks_sql = ", chunks_embedded = 0" if reset_chunks else ""
            conn.execute(
                f"""
                UPDATE kb_ingest_state
                SET status = 'pending', last_error = NULL,
                    started_at = NULL, finished_at = NULL, updated_at = ?{chunks_sql}
                WHERE file_path = ?
                """,
                (_utc_now(), file_path),
            )
        return True

    def list_running_files(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM kb_ingest_state
                WHERE status = 'running'
                ORDER BY updated_at
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_failed_files(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM kb_ingest_state
                WHERE status = 'failed'
                ORDER BY file_name
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_file_states(self, *, order: str = "file_name") -> list[dict[str, Any]]:
        if order == "updated_desc":
            order_clause = "ORDER BY (updated_at IS NULL), updated_at DESC"
        elif order == "file_name":
            order_clause = "ORDER BY file_name"
        else:
            raise ValueError(f"unsupported file state order: {order}")
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM kb_ingest_state {order_clause}"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_pending_files(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM kb_ingest_state
                WHERE status IN ('pending', 'queued')
                ORDER BY """ + _PRIORITY_ORDER_SQL + """, updated_at
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def claim_pending_file(self) -> dict[str, Any] | None:
        """Atomically claim one pending file (status -> running)."""
        if _sqlite_supports_returning():
            return self._claim_pending_file_returning()
        return self._claim_pending_file_legacy()

    def _claim_pending_file_returning(self) -> dict[str, Any] | None:
        now = _utc_now()
        with self._conn() as conn:
            row = conn.execute(
                """
                UPDATE kb_ingest_state
                SET status = 'running',
                    started_at = ?,
                    last_error = NULL,
                    updated_at = ?
                WHERE rowid = (
                    SELECT rowid FROM kb_ingest_state
                    WHERE status IN ('pending', 'queued')
                    ORDER BY """ + _PRIORITY_ORDER_SQL + """, updated_at
                    LIMIT 1
                )
                AND status IN ('pending', 'queued')
                RETURNING *
                """,
                (now, now),
            ).fetchone()
        return dict(row) if row else None

    def _claim_pending_file_legacy(self) -> dict[str, Any] | None:
        """BEGIN IMMEDIATE claim for SQLite < 3.35 (no UPDATE RETURNING)."""
        now = _utc_now()
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            target = conn.execute(
                """
                SELECT rowid FROM kb_ingest_state
                WHERE status IN ('pending', 'queued')
                ORDER BY """ + _PRIORITY_ORDER_SQL + """, updated_at
                LIMIT 1
                """
            ).fetchone()
            if target is None:
                return None
            rowid = int(target[0])
            conn.execute(
                """
                UPDATE kb_ingest_state
                SET status = 'running',
                    started_at = ?,
                    last_error = NULL,
                    updated_at = ?
                WHERE rowid = ? AND status IN ('pending', 'queued')
                """,
                (now, now, rowid),
            )
            row = conn.execute(
                "SELECT * FROM kb_ingest_state WHERE rowid = ?",
                (rowid,),
            ).fetchone()
        if row is None or row["status"] != "running":
            return None
        return dict(row)

    def create_job(self, job_id: str, job_type: str, message: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO ingest_jobs (id, job_type, status, message, created_at)
                VALUES (?, ?, 'queued', ?, ?)
                """,
                (job_id, job_type, message, _utc_now()),
            )

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        with self._conn() as conn:
            fields = dict(kwargs)
            if "finished_at" not in fields and fields.get("status") in ("done", "failed"):
                fields["finished_at"] = _utc_now()
            assignments = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(
                f"UPDATE ingest_jobs SET {assignments} WHERE id = ?",
                (*fields.values(), job_id),
            )

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM ingest_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
