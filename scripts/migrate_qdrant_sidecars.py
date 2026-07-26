#!/usr/bin/env python3
"""One-time migrate: rebuild TurboVec + BM25 from an existing Qdrant collection.

Safe to re-run: skips steps that are already caught up or not configured.
Does not change DENSE_BACKEND. Prints a message for each step (job log).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import httpx

from ingest.sidecar_lifecycle import ensure_sparse_sidecar, ensure_turbovec_sidecar
from ingest.sidecar_migrate import (
    health_count,
    health_max_points,
    needs_sidecar_rebuild,
    sparse_target_docs,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _timeout_s() -> float:
    raw = os.getenv("MIGRATE_HTTP_TIMEOUT_SEC", "7200").strip()
    try:
        return max(60.0, float(raw))
    except ValueError:
        return 7200.0


def _qdrant_points(qdrant_url: str, collection: str, client: httpx.Client) -> int:
    url = f"{qdrant_url.rstrip('/')}/collections/{collection}"
    response = client.get(url)
    response.raise_for_status()
    return int(response.json()["result"]["points_count"])


def _get_health(base_url: str, client: httpx.Client) -> dict[str, Any] | None:
    try:
        response = client.get(f"{base_url.rstrip('/')}/health")
        if response.status_code >= 400:
            return None
        body = response.json()
        return body if isinstance(body, dict) else None
    except Exception:
        return None


def _post_reindex(base_url: str, collection: str, client: httpx.Client) -> dict[str, Any]:
    response = client.post(
        f"{base_url.rstrip('/')}/reindex",
        json={"collection": collection},
    )
    response.raise_for_status()
    body = response.json()
    return body if isinstance(body, dict) else {}


def _post_save(base_url: str, client: httpx.Client) -> None:
    response = client.post(f"{base_url.rstrip('/')}/save")
    response.raise_for_status()


def run_migration(
    *,
    qdrant_url: str,
    collection: str,
    turbovec_url: str,
    sparse_url: str,
) -> int:
    """Return process exit code (0 = ok / nothing to do)."""
    timeout = _timeout_s()
    _log(f"Migration start collection={collection} timeout={int(timeout)}s")
    _log("Note: DENSE_BACKEND is not changed; flip to turbovec manually after TurboVec is full.")

    failures = 0
    with httpx.Client(timeout=timeout) as client:
        try:
            points = _qdrant_points(qdrant_url, collection, client)
        except Exception as exc:
            _log(f"[1/3] Qdrant: FAILED to read points ({exc})")
            return 1
        _log(f"[1/3] Qdrant: {points:,} points in {collection}")

        if points <= 0:
            _log("[1/3] Qdrant collection is empty — sidecars will be flushed to match.")

        # --- TurboVec ---
        tv = turbovec_url.strip()
        if not tv:
            _log("[2/3] TurboVec: not configured (TURBOVEC_URL empty) — skip.")
        else:
            _log("[2/3] TurboVec: ensuring sidecar is up…")
            ensure_turbovec_sidecar(tv, wait_health=True)
            health = _get_health(tv, client)
            vectors = health_count(health, "vectors", "docs")
            if health is None:
                _log("[2/3] TurboVec: health probe failed before rebuild — will try reindex anyway.")
            elif not needs_sidecar_rebuild(vectors, points):
                _log(
                    f"[2/3] TurboVec: already synced ({vectors:,} vectors = {points:,} points) — skip."
                )
            else:
                _log(
                    f"[2/3] TurboVec: rebuilding from Qdrant "
                    f"({vectors:,} vectors → target {points:,})…"
                )
                try:
                    result = _post_reindex(tv, collection, client)
                    after = health_count(result, "vectors", "docs")
                    _log(f"[2/3] TurboVec: reindex finished ({after:,} vectors).")
                    try:
                        _post_save(tv, client)
                        _log("[2/3] TurboVec: index saved to disk.")
                    except Exception as exc:
                        _log(f"[2/3] TurboVec: save skipped/failed ({exc}) — index may still be in memory.")
                except Exception as exc:
                    _log(f"[2/3] TurboVec: FAILED ({exc})")
                    failures += 1

        # --- BM25 ---
        sparse = sparse_url.strip()
        if not sparse:
            _log("[3/3] BM25: not configured (SPARSE_INDEX_URL empty) — skip.")
        else:
            _log("[3/3] BM25: ensuring sidecar is up…")
            ensure_sparse_sidecar(sparse, wait_health=True)
            health = _get_health(sparse, client)
            docs = health_count(health, "docs")
            max_points = health_max_points(health)
            target = sparse_target_docs(points, max_points=max_points)
            cap_note = (
                f" (capped at SPARSE_MAX_POINTS={max_points:,})"
                if max_points and max_points > 0
                else ""
            )
            if health is None:
                _log("[3/3] BM25: health probe failed before rebuild — will try reindex anyway.")
                need = True
            else:
                need = needs_sidecar_rebuild(docs, target)
            if not need:
                _log(
                    f"[3/3] BM25: already synced ({docs:,} docs = target {target:,}{cap_note}) — skip."
                )
            else:
                _log(
                    f"[3/3] BM25: rebuilding from Qdrant "
                    f"({docs:,} docs → target {target:,}{cap_note})…"
                )
                try:
                    result = _post_reindex(sparse, collection, client)
                    after = health_count(result, "docs")
                    _log(f"[3/3] BM25: reindex finished ({after:,} docs).")
                except Exception as exc:
                    _log(f"[3/3] BM25: FAILED ({exc})")
                    failures += 1

    if failures:
        _log(f"Migration finished with {failures} failure(s).")
        return 1
    _log("Migration complete.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "nomad_knowledge_base"),
    )
    parser.add_argument("--turbovec-url", default=os.getenv("TURBOVEC_URL", ""))
    parser.add_argument("--sparse-url", default=os.getenv("SPARSE_INDEX_URL", ""))
    args = parser.parse_args(argv)
    return run_migration(
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        turbovec_url=args.turbovec_url,
        sparse_url=args.sparse_url,
    )


if __name__ == "__main__":
    sys.exit(main())
