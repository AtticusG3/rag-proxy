"""Tests for MCP personal agent memory store (separate from curated KB)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MCP_RAG = REPO_ROOT / "sidecars" / "mcp_rag"
sys.path.insert(0, str(MCP_RAG))

from personal_store import (  # noqa: E402
    forget_note,
    format_notes_for_agent,
    recall_notes,
    store_note,
)


def test_store_recall_and_forget_are_isolated_to_personal_path(tmp_path: Path) -> None:
    """Personal notes must round-trip in the agent store, not imply KB writes."""
    store = tmp_path / "personal.sqlite"
    note = store_note(
        "pref.editor",
        "Use dark theme and pytest",
        tags="prefs coding",
        path=store,
    )
    assert note.key == "pref.editor"

    hits = recall_notes("pytest", top_k=5, path=store)
    assert len(hits) == 1
    assert hits[0].text.startswith("Use dark theme")
    assert "personal note" in format_notes_for_agent(hits).lower()

    assert forget_note("pref.editor", path=store) is True
    assert recall_notes("pytest", path=store) == []


def test_recall_ranks_token_overlap_over_unrelated(tmp_path: Path) -> None:
    """Recall should prefer notes that share query tokens so agents get the right memory."""
    store = tmp_path / "personal.sqlite"
    store_note("a", "homelab nginx reverse proxy", tags="", path=store)
    store_note("b", "favorite pizza toppings", tags="", path=store)

    hits = recall_notes("nginx proxy", top_k=5, path=store)
    assert hits[0].key == "a"
