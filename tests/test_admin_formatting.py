"""Tests for admin datetime formatting and file state ordering."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from rag_admin.db import AdminDatabase
from rag_admin.formatting import format_datetime


class TestAdminFormatting(unittest.TestCase):
    def test_format_datetime_iso(self) -> None:
        self.assertEqual(
            format_datetime("2026-06-25T12:13:25.083875+00:00"),
            "25-Jun-2026 12:13:25",
        )

    def test_format_datetime_empty(self) -> None:
        self.assertEqual(format_datetime(None), "")
        self.assertEqual(format_datetime(""), "")


class TestFileStateOrdering(unittest.TestCase):
    def test_list_file_states_updated_desc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "admin.sqlite")
            db = AdminDatabase(db_path)
            stamps = {
                "/zim/a.zim": "2026-06-25T10:00:00+00:00",
                "/zim/b.zim": "2026-06-25T12:00:00+00:00",
                "/zim/c.zim": "2026-06-25T11:00:00+00:00",
            }
            for path in stamps:
                db.upsert_file_state(path, status="indexed", file_type="zim")

            conn = sqlite3.connect(db_path)
            for path, ts in stamps.items():
                conn.execute(
                    "UPDATE kb_ingest_state SET updated_at = ? WHERE file_path = ?",
                    (ts, path),
                )
            conn.commit()
            conn.close()

            ordered = db.list_file_states(order="updated_desc")
            self.assertEqual(
                [row["file_name"] for row in ordered],
                ["b.zim", "c.zim", "a.zim"],
            )


if __name__ == "__main__":
    unittest.main()
