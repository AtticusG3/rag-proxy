#!/usr/bin/env python3
"""Chunk and ingest Project Gutenberg ebooks into Qdrant.

Usage:
    python ingest_books.py
"""
from __future__ import annotations

import os
import re
import sys

from ingest.chunking import chunk_text
from ingest.embedder import embed_texts
from ingest.qdrant_writer import (
    build_point,
    ensure_collection,
    get_collection_count,
    upsert_points,
)

BOOKS_DIR = os.getenv("BOOKS_DIR", "/home/kevyn/rag-proxy-dev/books")
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_proxy_dev")
EMBED_URL = os.getenv("EMBED_URL", "http://127.0.0.1:8089")
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "32"))


def strip_gutenberg_boilerplate(text: str) -> str:
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx >= 0:
            nl = text.find("\n", idx)
            start_idx = nl + 1 if nl >= 0 else idx + len(marker)
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx >= 0:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def main() -> None:
    ensure_collection(QDRANT_URL, COLLECTION)
    book_files = sorted(
        f
        for f in os.listdir(BOOKS_DIR)
        if f.endswith(".txt") and not f.startswith("1234") and not f.startswith("26253")
    )

    print(f"Found {len(book_files)} books in {BOOKS_DIR}")
    total_ingested = 0

    for book_file in book_files:
        book_path = os.path.join(BOOKS_DIR, book_file)
        book_name = book_file.replace(".txt", "").replace("_", " ").title()
        print(f"=== {book_name} ===")

        with open(book_path, encoding="utf-8", errors="replace") as handle:
            raw = handle.read()
        text = strip_gutenberg_boilerplate(raw)
        if len(text) < 100:
            print("  SKIP: too short after cleaning")
            continue

        chunks = chunk_text(text)
        print(f"  Chunks: {len(chunks)}")

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start : batch_start + BATCH_SIZE]
            embeddings = embed_texts(batch, embed_url=EMBED_URL)
            points = []
            for i, chunk in enumerate(batch):
                chunk_idx = batch_start + i
                points.append(
                    build_point(
                        text=chunk,
                        source=book_file,
                        title=book_name,
                        chunk_idx=chunk_idx,
                        embedding=embeddings[i],
                    )
                )
            upsert_points(QDRANT_URL, COLLECTION, points)
            total_ingested += len(points)

    count = get_collection_count(QDRANT_URL, COLLECTION)
    print(f"Total points ingested: {total_ingested}")
    print(f"Qdrant collection count: {count}")


if __name__ == "__main__":
    main()
