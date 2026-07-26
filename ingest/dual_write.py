"""Qdrant + TurboVec dual-write helpers (ingest delete path)."""

from __future__ import annotations

from ingest import turbovec_client
from ingest.qdrant_writer import delete_by_source, list_point_ids_by_source


def delete_source_points(
    qdrant_url: str,
    collection: str,
    source: str,
    *,
    turbovec_url: str | None = None,
) -> None:
    """List ids, delete from Qdrant, then best-effort TurboVec /remove."""
    ids = list_point_ids_by_source(qdrant_url, collection, source)
    delete_by_source(qdrant_url, collection, source)
    tv_url = turbovec_client.turbovec_url_from_env(turbovec_url)
    if tv_url and ids:
        turbovec_client.remove_ids(tv_url, ids)
