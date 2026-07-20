"""Bulk subscription removal must clear catalog rows and indexed local files."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from rag_admin.routes.explorer import delete_subscriptions_and_files


def test_bulk_delete_removes_each_subscription_and_indexed_file(tmp_path: Path) -> None:
    """Operators can remove many subscriptions at once; each local path is fully purged."""
    existing = tmp_path / "a.zim"
    existing.write_text("zim", encoding="utf-8")
    missing = tmp_path / "gone.zim"

    rows = {
        1: {"id": 1, "local_path": str(existing)},
        2: {"id": 2, "local_path": str(missing)},
        3: {"id": 3, "local_path": None},
    }
    db = MagicMock()
    db.delete_subscription.side_effect = lambda sub_id: rows.get(sub_id)
    worker = MagicMock()

    removed = delete_subscriptions_and_files(db, worker, [1, 2, 3, 99])

    assert removed == 3
    assert db.delete_subscription.call_count == 4
    # Purge even when the file is already gone so Qdrant/BM25/MemGraphRAG stay clean.
    assert worker.remove_file_from_index.call_args_list == [
        ((str(existing),),),
        ((str(missing),),),
    ]


def test_bulk_delete_with_no_ids_is_a_no_op() -> None:
    """Empty bulk submit must not touch the catalog or index."""
    db = MagicMock()
    worker = MagicMock()

    removed = delete_subscriptions_and_files(db, worker, [])

    assert removed == 0
    db.delete_subscription.assert_not_called()
    worker.remove_file_from_index.assert_not_called()
