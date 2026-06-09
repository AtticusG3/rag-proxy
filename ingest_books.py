#!/usr/bin/env python3
"""
ingest_books.py — chunk and ingest Project Gutenberg ebooks into Qdrant for rag-proxy dev.

Usage:
    cd /home/kevyn/code-projects/rag-proxy
    .venv/bin/python ingest_books.py
"""
import json
import os
import re
import sys
import time
import hashlib
import requests

# --- Config ---
BOOKS_DIR = "/home/kevyn/rag-proxy-dev/books"
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "rag_proxy_dev"
EMBED_URL = "http://127.0.0.1:8089"
EMBED_MODEL = "nomic-embed-text-v1.5"
CHUNK_SIZE = 512       # characters per chunk
CHUNK_OVERLAP = 64     # overlap between chunks
BATCH_SIZE = 32        # embed batch size

# --- Helpers ---

def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Gutenberg header/footer boilerplate."""
    # Find the actual book content
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
            # Find the end of this line
            nl = text.find('\n', idx)
            start_idx = nl + 1 if nl >= 0 else idx + len(marker)
            break
    
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx >= 0:
            end_idx = idx
            break
    
    return text[start_idx:end_idx].strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks at paragraph boundaries."""
    # Normalize whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    chunks = []
    # Split on paragraph boundaries
    paragraphs = text.split('\n\n')
    
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph would exceed size, flush current
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            # Keep overlap
            if len(current) > overlap:
                current = current[-overlap:] + "\n\n" + para
            else:
                current = para
        else:
            current = current + "\n\n" + para if current else para
    
    if current:
        chunks.append(current)
    
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call embed server to get embeddings."""
    resp = requests.post(
        f"{EMBED_URL}/v1/embeddings",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]


def get_qdrant_count() -> int:
    """Get current point count in collection."""
    resp = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}")
    resp.raise_for_status()
    return resp.json()["result"]["points_count"]


def upsert_points(points: list[dict]):
    """Upsert points into Qdrant."""
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        json={"points": points},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def make_point_id(text: str, source: str, chunk_idx: int) -> str:
    """Generate a deterministic point ID."""
    h = hashlib.md5(f"{source}:{chunk_idx}:{text[:100]}".encode()).hexdigest()
    return h


# --- Main ---

def main():
    book_files = sorted([
        f for f in os.listdir(BOOKS_DIR)
        if f.endswith('.txt') and not f.startswith('1234') and not f.startswith('26253')
    ])
    
    print(f"Found {len(book_files)} books in {BOOKS_DIR}")
    print(f"Qdrant: {QDRANT_URL}, collection: {COLLECTION}")
    print(f"Embed: {EMBED_URL}/v1/embeddings, model: {EMBED_MODEL}")
    print()
    
    total_chunks = 0
    total_ingested = 0
    
    for book_file in book_files:
        book_path = os.path.join(BOOKS_DIR, book_file)
        book_name = book_file.replace('.txt', '').replace('_', ' ').title()
        
        print(f"=== {book_name} ===")
        
        # Read and clean
        with open(book_path, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()
        
        text = strip_gutenberg_boilerplate(raw)
        print(f"  Raw: {len(raw):,} chars -> Clean: {len(text):,} chars")
        
        if len(text) < 100:
            print(f"  SKIP: too short after cleaning")
            continue
        
        # Chunk
        chunks = chunk_text(text)
        print(f"  Chunks: {len(chunks)}")
        total_chunks += len(chunks)
        
        # Embed and ingest in batches
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start:batch_start + BATCH_SIZE]
            
            try:
                embeddings = embed_texts(batch)
            except Exception as e:
                print(f"  EMBED ERROR: {e}")
                time.sleep(2)
                try:
                    embeddings = embed_texts(batch)
                except Exception as e2:
                    print(f"  EMBED RETRY FAILED: {e2}")
                    continue
            
            points = []
            for i, (chunk_text_content, embedding) in enumerate(zip(batch, embeddings)):
                chunk_idx = batch_start + i
                point_id = make_point_id(chunk_text_content, book_file, chunk_idx)
                points.append({
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "text": chunk_text_content[:2000],  # truncate for storage
                        "source": book_file,
                        "book": book_name,
                        "chunk_idx": chunk_idx,
                        "chunk_size": len(chunk_text_content),
                    }
                })
            
            try:
                upsert_points(points)
                total_ingested += len(points)
            except Exception as e:
                print(f"  UPSERT ERROR: {e}")
            
            if (batch_start // BATCH_SIZE) % 5 == 0:
                print(f"  Progress: {batch_start + len(batch)}/{len(chunks)} chunks")
        
        print(f"  Done: {len(chunks)} chunks ingested")
        print()
    
    # Final count
    count = get_qdrant_count()
    print(f"=== Summary ===")
    print(f"Total chunks created: {total_chunks}")
    print(f"Total points ingested: {total_ingested}")
    print(f"Qdrant collection count: {count}")


if __name__ == "__main__":
    main()
