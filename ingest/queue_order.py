"""Queue claim ordering for the ingest worker.

Priority band (high -> mid -> low) always wins, then the persisted
queue_sort / queue_dir, then FIFO (updated_at). The Jobs column sort is
display-only. "Switch to top of queue" writes the visible sort into this
order and preempts running files. Status polls must not write it.
"""

from __future__ import annotations

import os
from typing import Any

QUEUE_SORT_KEYS = ("name", "priority", "status", "size", "updated")
DEFAULT_QUEUE_SORT = "updated"
DEFAULT_QUEUE_DIR = "asc"

_PRIORITY_RANK = {"high": 0, "mid": 1, "low": 2}


def _row_size(row: dict[str, Any]) -> int:
    if row.get("file_size") is not None:
        return int(row["file_size"])
    try:
        return os.path.getsize(str(row.get("file_path") or ""))
    except OSError:
        return 0


def order_queue_rows(
    rows: list[dict[str, Any]],
    *,
    sort: str,
    direction: str,
) -> list[dict[str, Any]]:
    """Order pending rows for claiming: priority band, then user sort, then FIFO."""
    if sort not in QUEUE_SORT_KEYS:
        sort = DEFAULT_QUEUE_SORT
    reverse = direction == "desc"

    def user_key(row: dict[str, Any]) -> Any:
        if sort == "name":
            return (row.get("file_name") or "").lower()
        if sort == "size":
            return _row_size(row)
        if sort == "updated":
            return str(row.get("updated_at") or "")
        if sort == "status":
            return str(row.get("display_status") or row.get("status") or "")
        # A priority sort has no secondary order within a priority band.
        return 0

    ordered = sorted(rows, key=lambda r: str(r.get("updated_at") or ""))
    ordered = sorted(ordered, key=user_key, reverse=reverse)
    return sorted(
        ordered,
        key=lambda r: _PRIORITY_RANK.get(r.get("priority") or "mid", 1),
    )
