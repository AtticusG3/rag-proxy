"""Test MemGraphRAG retrieval end-to-end against the built index.

Bypasses rag_proxy.config by hardcoding the service URLs we need.
Loads the SQLite index built by build_memgraphrag_index.py, runs a
few test queries, and prints the top passages returned by PPR.

Usage:
    .venv/bin/python scripts/test_memgraphrag_retrieve.py \\
        --index /tmp/memgraph_200.sqlite \\
        --queries "what is photosynthesis" "how does SSL work" "who founded Wikipedia"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import httpx
import numpy as np

# Make the rag_proxy package importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_proxy.memgraphrag.memory import ThreeLayerMemory

log = logging.getLogger("memgraphrag.test_retrieve")


# ---------------------------------------------------------------------------
# Minimal embed client (nomic-embed on llama.cpp)
# ---------------------------------------------------------------------------

async def embed_text(url: str, text: str) -> list[float]:
    """Embed text via nomic-embed endpoint."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json={"input": text})
        resp.raise_for_status()
        data = resp.json()
        if "data" in data:
            return data["data"][0]["embedding"]
        return data["embedding"]


# ---------------------------------------------------------------------------
# Minimal rerank client (cross-encoder sidecar)
# ---------------------------------------------------------------------------

async def rerank_texts(rerank_url: str, query: str, texts: list[str]) -> list[float]:
    """Rerank texts against query via cross-encoder sidecar."""
    if not texts:
        return []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(rerank_url, json={"query": query, "passages": texts})
            resp.raise_for_status()
            return resp.json().get("scores", [1.0] * len(texts))
    except Exception as e:
        log.warning("Reranker failed: %s, using uniform scores", e)
        return [1.0] * len(texts)


# ---------------------------------------------------------------------------
# PPR (Personalized PageRank) — simplified from retrieval.py
# ---------------------------------------------------------------------------

def ppr(
    seed_scores: dict[int, float],
    adj: dict[int, list[int]],
    damping: float = 0.85,
    iterations: int = 20,
    threshold: float = 0.01,
) -> dict[int, float]:
    """Personalized PageRank on the fact graph."""
    if not seed_scores:
        return {}
    nodes = list(adj.keys())
    n = len(nodes)
    if n == 0:
        return {}

    node_to_local = {node: i for i, node in enumerate(nodes)}
    teleport = np.zeros(n)
    total_seed = sum(seed_scores.values())
    if total_seed == 0:
        return {}
    for fi, score in seed_scores.items():
        if fi in node_to_local:
            teleport[node_to_local[fi]] = score / total_seed

    scores = teleport.copy()
    for _ in range(iterations):
        new_scores = (1 - damping) * teleport
        for i, node in enumerate(nodes):
            neighbors = adj.get(node, [])
            if not neighbors:
                new_scores += scores[i] / n
                continue
            share = damping * scores[i] / len(neighbors)
            for nb in neighbors:
                if nb in node_to_local:
                    new_scores[node_to_local[nb]] += share
        scores = new_scores

    result: dict[int, float] = {}
    for i, node in enumerate(nodes):
        if scores[i] > threshold:
            result[node] = float(scores[i])
    return result


# ---------------------------------------------------------------------------
# Build fact adjacency from memory
# ---------------------------------------------------------------------------

def build_fact_adj(memory: ThreeLayerMemory) -> dict[int, list[int]]:
    """Build fact-to-fact adjacency from inter-layer connections."""
    adj: dict[int, list[int]] = {}
    for fi in memory.facts:
        related: set[int] = set()
        node = memory.facts[fi]
        # same schema
        if node.schema_idx in memory.schemas:
            for rfi in memory.schemas[node.schema_idx].fact_indices:
                if rfi != fi:
                    related.add(rfi)
        # same passages
        for pi in node.passage_indices:
            if pi in memory.passages:
                for rfi in memory.passages[pi].fact_indices:
                    if rfi != fi:
                        related.add(rfi)
        adj[fi] = list(related)
    return adj


# ---------------------------------------------------------------------------
# Full retrieval pipeline
# ---------------------------------------------------------------------------

async def retrieve(
    memory: ThreeLayerMemory,
    adj: dict[int, list[int]],
    embed_url: str,
    rerank_url: str | None,
    query: str,
    top_k: int = 5,
    fact_top_k: int = 20,
) -> list[dict]:
    """Full MemGraphRAG retrieval: embed → score facts → rerank → PPR → passages."""
    t0 = time.time()

    # 1. Embed query
    query_emb = np.array(await embed_text(embed_url, query), dtype=np.float32)
    query_norm = np.linalg.norm(query_emb)
    if query_norm == 0:
        return []

    # 2. Score facts
    scored: list[tuple[int, float]] = []
    for fi, fact in memory.facts.items():
        fact_text = fact.triple_str
        try:
            fact_emb = np.array(await embed_text(embed_url, fact_text), dtype=np.float32)
            fact_norm = np.linalg.norm(fact_emb)
            if fact_norm == 0:
                continue
            score = float(np.dot(query_emb, fact_emb) / (query_norm * fact_norm))
            scored.append((fi, score))
        except Exception as e:
            log.debug("Failed to score fact %d: %s", fi, e)

    scored.sort(key=lambda x: x[1], reverse=True)
    log.info("Scored %d facts in %.1fs", len(scored), time.time() - t0)

    # 3. Rerank top facts
    top_indices = [fi for fi, _ in scored[:fact_top_k]]
    top_texts = [memory.facts[fi].triple_str for fi in top_indices if fi in memory.facts]
    scores = await rerank_texts(rerank_url, query, top_texts) if rerank_url else [s for _, s in scored[:fact_top_k]]
    reranked = list(zip(top_indices, scores))
    reranked.sort(key=lambda x: x[1], reverse=True)

    # 4. PPR
    seed_scores = {fi: score for fi, score in reranked if score > 0}
    ppr_scores = ppr(seed_scores, adj)
    log.info("PPR: %d facts above threshold", len(ppr_scores))

    # 5. Aggregate to passages
    passage_scores: dict[int, float] = {}
    for fi, score in ppr_scores.items():
        if fi not in memory.facts:
            continue
        fact = memory.facts[fi]
        for pi in fact.passage_indices:
            if pi in memory.passages:
                passage_scores[pi] = passage_scores.get(pi, 0.0) + score * 0.5

    ranked = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)

    # 6. Build results
    results: list[dict] = []
    for pi, score in ranked[:top_k]:
        if pi in memory.passages:
            p = memory.passages[pi]
            results.append({
                "chunk_id": p.chunk_id,
                "score": round(score, 4),
                "text_preview": p.content[:300],
                "fact_count": len(p.fact_indices),
            })

    log.info("Returning %d passages in %.1fs total", len(results), time.time() - t0)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Test MemGraphRAG retrieval")
    parser.add_argument("--index", required=True, help="Path to SQLite index")
    parser.add_argument("--embed-url", default="http://192.168.1.202:8089/v1/embeddings",
                        help="Embedding endpoint URL")
    parser.add_argument("--rerank-url", default="http://192.168.1.202:8095/rerank",
                        help="Reranker endpoint URL (optional)")
    parser.add_argument("--queries", nargs="+", default=[
        "what is the structure of DNA",
        "how does public key cryptography work",
        "who founded the Wikimedia Foundation",
    ], help="Test queries")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Load memory
    log.info("Loading index from %s ...", args.index)
    memory = ThreeLayerMemory(db_path=args.index)
    log.info("Loaded: %s", memory.stats)

    if not memory.facts:
        log.error("No facts in index — MemGraphRAG retrieval needs entity-relation triples. "
                  "Rebuild without --skip-relations.")
        sys.exit(1)

    # Build adjacency
    adj = build_fact_adj(memory)
    log.info("Fact graph: %d nodes", len(adj))

    # Run queries
    for query in args.queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        results = await retrieve(
            memory, adj, args.embed_url, args.rerank_url,
            query, top_k=args.top_k,
        )
        if not results:
            print("  (no results)")
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] score={r['score']} facts={r['fact_count']} id={r['chunk_id'][:20]}")
            print(f"      {r['text_preview'][:200]}")


if __name__ == "__main__":
    asyncio.run(main())
