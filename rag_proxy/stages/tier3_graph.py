"""Knowledge graph lookup (SQLite)."""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, IntentLabel, RequestContext

log = logging.getLogger("rag-proxy")

_ENTITY = re.compile(
    r"\b(nomad|qdrant|llama-swap|nomic-embed|docker|openmediavault|omv)\b",
    re.I,
)


def _ensure_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                kind TEXT,
                name TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                src TEXT,
                dst TEXT,
                rel TEXT,
                PRIMARY KEY (src, dst, rel)
            )
            """
        )


def _query_graph(db_path: Path, seeds: list[str], max_depth: int) -> list[str]:
    if not db_path.exists():
        return []
    lines: list[str] = []
    try:
        with sqlite3.connect(db_path) as conn:
            for seed in seeds[:5]:
                rows = conn.execute(
                    """
                    SELECT e.rel, d.name, d.kind
                    FROM edges e
                    JOIN entities d ON e.dst = d.id
                    JOIN entities s ON e.src = s.id
                    WHERE lower(s.name) LIKE ?
                    LIMIT 10
                    """,
                    (f"%{seed.lower()}%",),
                ).fetchall()
                for rel, name, kind in rows:
                    lines.append(f"- {seed} -[{rel}]-> {kind}:{name}")
    except Exception as e:
        log.warning(f"Graph query failed: {e}")
    return lines


async def run_graph(ctx: RequestContext) -> None:
    if not settings.enable_graph_lookup or not ctx.query_text:
        return
    if ctx.intent not in (
        IntentLabel.INFRA_DEBUG,
        IntentLabel.TROUBLESHOOTING,
        IntentLabel.LOG_ANALYSIS,
    ):
        return

    seeds = list({m.group(0).lower() for m in _ENTITY.finditer(ctx.query_text)})
    if not seeds:
        return

    db_path = Path(settings.graph_db_path)
    try:
        _ensure_schema(db_path)
        lines = _query_graph(db_path, seeds, settings.graph_max_depth)
        if lines:
            text = "Infrastructure graph context:\n" + "\n".join(lines)
            ctx.hits.append(ChunkHit(id="graph", text=text, score=0.9, source="graph"))
            ctx.chunk_texts.append(text)
            ctx.stage_trace.append(f"graph:{len(lines)}")
    except Exception as e:
        ctx.errors.append(f"graph:{e}")
