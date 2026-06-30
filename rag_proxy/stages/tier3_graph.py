"""Knowledge graph lookup (SQLite)."""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, RequestContext

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


def _seed_frontier(conn: sqlite3.Connection, seed: str, limit: int) -> set[str]:
    rows = conn.execute(
        """
        SELECT id
        FROM entities
        WHERE lower(name) LIKE ?
        LIMIT ?
        """,
        (f"%{seed.lower()}%", limit),
    ).fetchall()
    return {row[0] for row in rows}


def _query_graph(db_path: Path, seeds: list[str], max_depth: int) -> list[str]:
    if not db_path.exists():
        return []
    depth = max(1, max_depth)
    lines: list[str] = []
    try:
        with sqlite3.connect(db_path) as conn:
            for seed in seeds[:5]:
                frontier = _seed_frontier(conn, seed, limit=10)
                visited_nodes: set[str] = set(frontier)
                seen_edges: set[tuple[str, str, str]] = set()
                for _ in range(depth):
                    if not frontier:
                        break
                    placeholders = ",".join("?" for _ in frontier)
                    rows = conn.execute(
                        f"""
                        SELECT e.src, e.dst, e.rel, s.name, d.name, d.kind
                        FROM edges e
                        JOIN entities s ON e.src = s.id
                        JOIN entities d ON e.dst = d.id
                        WHERE e.src IN ({placeholders})
                        LIMIT 50
                        """,
                        tuple(frontier),
                    ).fetchall()
                    if not rows:
                        break
                    next_frontier: set[str] = set()
                    for src, dst, rel, src_name, dst_name, dst_kind in rows:
                        edge_key = (src, dst, rel)
                        if edge_key in seen_edges:
                            continue
                        seen_edges.add(edge_key)
                        lines.append(f"- {src_name} -[{rel}]-> {dst_kind}:{dst_name}")
                        if dst not in visited_nodes:
                            next_frontier.add(dst)
                            visited_nodes.add(dst)
                    frontier = next_frontier
    except Exception as e:
        log.warning(f"Graph query failed: {e}")
    return lines


async def run_graph(ctx: RequestContext) -> None:
    if not ctx.query_text:
        return

    seeds = list({m.group(0).lower() for m in _ENTITY.finditer(ctx.query_text)})
    if not seeds:
        return

    db_path = Path(settings.graph_db_path)
    try:
        lines = _query_graph(db_path, seeds, settings.graph_max_depth)
        if lines:
            text = "Infrastructure graph context:\n" + "\n".join(lines)
            ctx.hits.append(ChunkHit(id="graph", text=text, score=0.9, source="graph"))
            ctx.stage_trace.append(f"graph:{len(lines)}")
    except Exception as e:
        ctx.errors.append(f"graph:{e}")
