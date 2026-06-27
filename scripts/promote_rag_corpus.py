"""Promote captured RAG improvement records into a derived Qdrant collection.

Examples:
  python scripts/promote_rag_corpus.py \
    --input /var/lib/rag_proxy/capture/rag_improvement.jsonl \
    --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from rag_proxy.config import settings
from rag_proxy.rag_corpus_promoter import promote_rag_record, should_promote_rag_record


def load_records(path: Path, *, since: str = "", limit: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("record_type") != "rag_turn":
                continue
            if since and str(record.get("ts") or "") < since:
                continue
            records.append(record)
            if limit and len(records) >= limit:
                break
    return records


async def promote_records(records: list[dict[str, Any]], *, dry_run: bool) -> tuple[int, int]:
    eligible = 0
    promoted = 0
    for record in records:
        if not should_promote_rag_record(record):
            continue
        eligible += 1
        if dry_run:
            continue
        if await promote_rag_record(record):
            promoted += 1
    return eligible, promoted


def apply_args(args: argparse.Namespace) -> None:
    settings.enable_rag_corpus_auto_ingest = True
    if args.qdrant_url:
        settings.qdrant_url = args.qdrant_url
    if args.collection:
        settings.rag_corpus_collection = args.collection
    if args.min_answer_chars is not None:
        settings.rag_corpus_min_answer_chars = args.min_answer_chars
    settings.rag_corpus_require_chunks = args.require_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote captured RAG Q&A into Qdrant.")
    parser.add_argument("--input", required=True, type=Path, help="Path to rag_improvement.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Count eligible records only")
    parser.add_argument("--since", default="", help="Minimum ISO timestamp, compared lexically")
    parser.add_argument("--limit", type=int, default=0, help="Maximum records to inspect")
    parser.add_argument("--qdrant-url", default="", help="Override QDRANT_URL")
    parser.add_argument("--collection", default="", help="Override RAG_CORPUS_COLLECTION")
    parser.add_argument(
        "--min-answer-chars",
        type=int,
        default=None,
        help="Override RAG_CORPUS_MIN_ANSWER_CHARS",
    )
    parser.add_argument(
        "--require-chunks",
        action="store_true",
        help="Only promote turns that injected at least one chunk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_args(args)
    records = load_records(args.input, since=args.since, limit=args.limit)
    eligible, promoted = asyncio.run(promote_records(records, dry_run=args.dry_run))
    if args.dry_run:
        print(f"Eligible {eligible} of {len(records)} inspected record(s)")
    else:
        print(f"Promoted {promoted} of {eligible} eligible record(s)")


if __name__ == "__main__":
    main()
