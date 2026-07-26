"""Personal agent memory (Hermes/OpenClaw-style local_store) — separate from curated KB."""

from __future__ import annotations

import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_DEFAULT_STORE = Path.home() / ".local" / "share" / "rag_proxy" / "mcp_personal.sqlite"


@dataclass(frozen=True)
class PersonalNote:
    key: str
    text: str
    tags: str
    updated_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "text": self.text,
            "tags": self.tags,
            "updated_at": self.updated_at,
        }


def default_store_path() -> Path:
    raw = os.getenv("MCP_PERSONAL_STORE_PATH", "").strip()
    if raw:
        return Path(raw)
    return _DEFAULT_STORE


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            key TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_notes_updated ON notes(updated_at DESC)"
    )
    return conn


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if t}


def store_note(
    key: str,
    text: str,
    *,
    tags: str = "",
    path: Path | None = None,
) -> PersonalNote:
    note_key = key.strip()
    body = text.strip()
    if not note_key:
        raise ValueError("key must be non-empty")
    if not body:
        raise ValueError("text must be non-empty")
    store = path or default_store_path()
    now = time.time()
    tag_str = tags.strip()
    with _connect(store) as conn:
        conn.execute(
            """
            INSERT INTO notes(key, text, tags, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                text = excluded.text,
                tags = excluded.tags,
                updated_at = excluded.updated_at
            """,
            (note_key, body, tag_str, now),
        )
        conn.commit()
    return PersonalNote(key=note_key, text=body, tags=tag_str, updated_at=now)


def forget_note(key: str, *, path: Path | None = None) -> bool:
    note_key = key.strip()
    if not note_key:
        return False
    store = path or default_store_path()
    with _connect(store) as conn:
        cur = conn.execute("DELETE FROM notes WHERE key = ?", (note_key,))
        conn.commit()
        return cur.rowcount > 0


def recall_notes(
    query: str,
    *,
    top_k: int = 5,
    path: Path | None = None,
) -> list[PersonalNote]:
    """Lexical recall: token overlap over personal notes only (not the curated KB)."""
    limit = max(1, min(int(top_k), 50))
    q = query.strip()
    store = path or default_store_path()
    with _connect(store) as conn:
        rows = conn.execute(
            "SELECT key, text, tags, updated_at FROM notes ORDER BY updated_at DESC"
        ).fetchall()

    if not rows:
        return []

    if not q:
        return [
            PersonalNote(
                key=str(r["key"]),
                text=str(r["text"]),
                tags=str(r["tags"] or ""),
                updated_at=float(r["updated_at"]),
            )
            for r in rows[:limit]
        ]

    q_tokens = _tokenize(q)
    scored: list[tuple[float, PersonalNote]] = []
    for row in rows:
        note = PersonalNote(
            key=str(row["key"]),
            text=str(row["text"]),
            tags=str(row["tags"] or ""),
            updated_at=float(row["updated_at"]),
        )
        hay = f"{note.key} {note.tags} {note.text}"
        if q.lower() in hay.lower():
            score = 1000.0 + len(q_tokens.intersection(_tokenize(hay)))
        else:
            overlap = len(q_tokens.intersection(_tokenize(hay)))
            if overlap == 0:
                continue
            score = float(overlap)
        scored.append((score, note))

    scored.sort(key=lambda item: (-item[0], -item[1].updated_at))
    return [note for _, note in scored[:limit]]


def format_notes_for_agent(notes: list[PersonalNote]) -> str:
    if not notes:
        return "No matching personal notes."
    lines = [f"Found {len(notes)} personal note(s):\n"]
    for idx, note in enumerate(notes, start=1):
        tag_bit = f" tags={note.tags}" if note.tags else ""
        lines.append(f"### [{idx}] `{note.key}`{tag_bit}\n\n{note.text}\n")
    return "\n".join(lines)
