"""Rolling memory stage: TTL on load and refresh guard."""

import asyncio
import sqlite3
import time

import pytest

from rag_proxy.config import settings
from rag_proxy.context import RequestContext
from rag_proxy.stages import tier3_memory


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "memory.sqlite"
    monkeypatch.setattr(settings, "enable_rolling_memory", True)
    monkeypatch.setattr(settings, "memory_db_path", str(path))
    monkeypatch.setattr(settings, "memory_ttl_hours", 72)
    monkeypatch.setattr(settings, "memory_refresh_turns", 8)
    tier3_memory._ensure_memory_schema(path)
    return path


def _ctx(turns: int = 1) -> RequestContext:
    ctx = RequestContext()
    ctx.conversation_id = "conv-1"
    ctx.query_text = "what is the test rig called?"
    ctx.messages = [{"role": "user", "content": "q"} for _ in range(turns)]
    return ctx


def _age(path, hours: float) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE session_memory SET updated_at = ?", (time.time() - hours * 3600,))


def test_fresh_row_is_injected(db):
    tier3_memory._save_summary(db, "conv-1", "old question", 8)
    ctx = _ctx()
    asyncio.run(tier3_memory.run_memory(ctx))
    assert "memory:loaded" in ctx.stage_trace
    assert "old question" in ctx.messages[0]["content"]


def test_expired_row_is_not_injected(db):
    tier3_memory._save_summary(db, "conv-1", "old question", 8)
    _age(db, 73)
    ctx = _ctx()
    asyncio.run(tier3_memory.run_memory(ctx))
    assert "memory:loaded" not in ctx.stage_trace
    assert len(ctx.messages) == 1


def test_zero_ttl_never_expires(db, monkeypatch):
    monkeypatch.setattr(settings, "memory_ttl_hours", 0)
    tier3_memory._save_summary(db, "conv-1", "old question", 8)
    _age(db, 10000)
    ctx = _ctx()
    asyncio.run(tier3_memory.run_memory(ctx))
    assert "memory:loaded" in ctx.stage_trace


def test_zero_refresh_turns_disables_refresh_without_errors(db, monkeypatch):
    monkeypatch.setattr(settings, "memory_refresh_turns", 0)
    ctx = _ctx(turns=8)
    asyncio.run(tier3_memory.run_memory(ctx))
    assert ctx.errors == []
    assert "memory:refreshed" not in ctx.stage_trace
    assert tier3_memory._load_summary(db, "conv-1") is None


def test_refresh_saves_question_on_nth_turn(db):
    ctx = _ctx(turns=8)
    asyncio.run(tier3_memory.run_memory(ctx))
    assert "memory:refreshed" in ctx.stage_trace
    assert tier3_memory._load_summary(db, "conv-1") == "what is the test rig called?"
