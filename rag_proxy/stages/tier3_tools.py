"""Tool-augmented retrieval (read-only, whitelisted)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, RequestContext, RetrievalDecision

log = logging.getLogger("rag-proxy")


def _path_allowed(path: Path, roots: list[str]) -> bool:
    try:
        resolved = path.resolve()
        for root in roots:
            root_path = Path(root).resolve()
            if resolved == root_path or root_path in resolved.parents:
                return True
    except OSError:
        return False
    return False


async def _read_file_limited(path: Path, max_chars: int) -> str:
    def _read() -> str:
        data = path.read_text(encoding="utf-8", errors="replace")
        return data[:max_chars]

    return await asyncio.wait_for(
        asyncio.to_thread(_read),
        timeout=settings.tool_timeout_sec,
    )


async def run_tools(ctx: RequestContext) -> None:
    if not settings.enable_tools or ctx.retrieval == RetrievalDecision.SKIP:
        return

    roots = settings.tool_roots()
    if not roots:
        ctx.errors.append("tools:no_roots")
        return

    query = ctx.effective_query() or ""
    added = 0
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for pattern in ("docker-compose.yml", "compose.yaml", ".env.example"):
            candidate = root_path / pattern
            if candidate.is_file() and _path_allowed(candidate, roots):
                try:
                    snippet = await _read_file_limited(
                        candidate, settings.tool_max_output_chars
                    )
                    label = f"tool:file:{candidate.name}"
                    ctx.hits.append(
                        ChunkHit(id=label, text=snippet, score=0.85, source="tool")
                    )
                    ctx.chunk_texts.append(f"[{candidate}]\n{snippet}")
                    added += 1
                    ctx.stage_trace.append(label)
                except Exception as e:
                    ctx.errors.append(f"tool:{e}")
        if added:
            break
