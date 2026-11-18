"""Rolling session memory (SQLite): recalls the last saved question per conversation."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from pathlib import Path

from rag_proxy.config import settings
from rag_proxy.context import RequestContext
from rag_proxy.rag_context import ROLLING_MEMORY_PREFIX, wrap_injected_block

log = logging.getLogger("rag-proxy")


def _ensure_memory_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_memory (
                conversation_id TEXT PRIMARY KEY,
                summary TEXT,
                turn_count INTEGER,
                updated_at REAL
            )
            """
        )


def _load_summary(db_path: Path, conversation_id: str) -> str | None:
    if not db_path.exists():
        return None
    # TTL <= 0 means rows never expire. Expired rows are skipped, not deleted;
    # the next refresh overwrites them.
    ttl = settings.memory_ttl_hours
    cutoff = time.time() - ttl * 3600 if ttl > 0 else 0
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT summary FROM session_memory WHERE conversation_id = ? AND updated_at >= ?",
            (conversation_id, cutoff),
        ).fetchone()
        return row[0] if row else None


def _save_summary(db_path: Path, conversation_id: str, summary: str, turns: int) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO session_memory (conversation_id, summary, turn_count, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                summary=excluded.summary,
                turn_count=excluded.turn_count,
                updated_at=excluded.updated_at
            """,
            (conversation_id, summary, turns, time.time()),
        )


async def run_memory(ctx: RequestContext) -> None:
    if not settings.enable_rolling_memory:
        return

    conv_id = ctx.conversation_id
    if not conv_id:
        return

    db_path = Path(settings.memory_db_path)
    try:
        await asyncio.to_thread(_ensure_memory_schema, db_path)
        summary = await asyncio.to_thread(_load_summary, db_path, conv_id)
        if summary:
            block = wrap_injected_block(ROLLING_MEMORY_PREFIX, summary)
            if ctx.messages and ctx.messages[0].get("role") == "system":
                ctx.messages[0] = {
                    **ctx.messages[0],
                    "content": block + "\n\n" + ctx.messages[0]["content"],
                }
            else:
                ctx.messages.insert(0, {"role": "system", "content": block})
            ctx.stage_trace.append("memory:loaded")

        user_turns = sum(1 for m in ctx.messages if m.get("role") == "user")
        refresh = settings.memory_refresh_turns
        if refresh > 0 and user_turns > 0 and user_turns % refresh == 0:
            # shortcut: stores the latest question verbatim, not a model summary, upgrade if real summaries are wanted
            brief = (ctx.query_text or "")[:500]
            await asyncio.to_thread(_save_summary, db_path, conv_id, brief, user_turns)
            ctx.stage_trace.append("memory:refreshed")
    except Exception as e:
        ctx.errors.append(f"memory:{e}")
