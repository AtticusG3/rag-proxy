"""MemGraphRAG retrieval: fact scoring -> rerank -> PPR graph walk -> passage retrieval.

Implements the memory-guided online retrieval from the MemGraphRAG paper:
  1. Embed query, score facts via dense similarity
  2. Rerank facts with cross-encoder
  3. Multi-layer memory filtering (schema -> fact)
  4. Structure-aware node initialization + Personalized PageRank on the memory graph
  5. Return top passages ranked by PPR score

When no facts score above zero similarity, returns an empty list (no dense fallback).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import httpx
import numpy as np

from rag_proxy.config import settings
from rag_proxy.context import ChunkHit
from rag_proxy.memgraphrag.memory import ThreeLayerMemory

log = logging.getLogger("rag-proxy.memgraphrag.retrieval")

_EMBED_MODEL = "nomic-embed-text-v1.5"


class MemGraphRetriever:
    """Online retrieval using the three-layer memory graph."""

    def __init__(
        self,
        memory: ThreeLayerMemory,
        embed_url: str | None = None,
        reranker_url: str | None = None,
        top_k: int = 5,
        fact_top_k: int = 20,
        ppr_damping: float = 0.85,
        ppr_iterations: int = 20,
        ppr_threshold: float = 0.01,
        passage_node_weight: float = 0.5,
    ):
        self.memory = memory
        self.embed_url = embed_url or settings.embed_url
        self.reranker_url = reranker_url or settings.reranker_url
        self.top_k = top_k
        self.fact_top_k = fact_top_k
        self.ppr_damping = ppr_damping
        self.ppr_iterations = ppr_iterations
        self.ppr_threshold = ppr_threshold
        self.passage_node_weight = passage_node_weight

        # Build adjacency for PPR (fact-to-fact graph)
        self._fact_adj: dict[int, list[int]] = {}
        self._build_fact_graph()

    def _build_fact_graph(self) -> None:
        """Build fact-to-fact adjacency from inter-layer connections."""
        for fi in self.memory.facts:
            self._fact_adj[fi] = self.memory.get_related_fact_indices(fi)
        log.info("Built fact graph: %d nodes", len(self._fact_adj))

    # -- embedding ---------------------------------------------------------

    async def _embed_query(self, query: str) -> list[float]:
        """Embed a query string via the nomic-embed OpenAI-compatible API."""
        trimmed = query.strip()
        if not trimmed:
            return []
        url = f"{self.embed_url.rstrip('/')}/v1/embeddings"
        payload = {"model": _EMBED_MODEL, "input": trimmed}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    # -- fact scoring ------------------------------------------------------

    async def score_facts(self, query: str) -> list[tuple[int, float]]:
        """Score all facts against the query using precomputed embedding cosine similarity.

        Query is embedded once via HTTP; fact vectors are loaded from memory (index build).
        Facts without stored embeddings are skipped.

        Returns list of (fact_idx, score) sorted by score descending.
        """
        query_emb = np.array(await self._embed_query(query), dtype=np.float32)
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return []

        scored: list[tuple[int, float]] = []
        for fi, fact in self.memory.facts.items():
            if not fact.embedding:
                continue
            fact_emb = np.array(fact.embedding, dtype=np.float32)
            fact_norm = np.linalg.norm(fact_emb)
            if fact_norm == 0:
                continue
            score = float(np.dot(query_emb, fact_emb) / (query_norm * fact_norm))
            scored.append((fi, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # -- reranking ---------------------------------------------------------

    async def rerank_facts(
        self, query: str, fact_indices: list[int], fact_texts: list[str]
    ) -> list[tuple[int, float]]:
        """Rerank facts using the cross-encoder sidecar.

        Returns list of (fact_idx, score) sorted by score descending.
        """
        if not fact_indices or not self.reranker_url:
            return list(zip(fact_indices, [1.0] * len(fact_indices)))

        pairs = [{"query": query, "document": text} for text in fact_texts]
        try:
            timeout = settings.rerank_timeout_ms / 1000.0 + 0.5
            url = f"{self.reranker_url.rstrip('/')}/rerank"
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url,
                    json={"pairs": pairs, "top_k": len(fact_indices)},
                )
                resp.raise_for_status()
                order = resp.json().get("indices", [])
                if not order:
                    return list(zip(fact_indices, [1.0] * len(fact_indices)))
                ranked: list[tuple[int, float]] = []
                for rank, pair_idx in enumerate(order):
                    if 0 <= pair_idx < len(fact_indices):
                        fi = fact_indices[pair_idx]
                        ranked.append((fi, float(len(fact_indices) - rank)))
                if len(ranked) != len(fact_indices):
                    log.warning(
                        "Reranker returned %d indices for %d facts",
                        len(ranked),
                        len(fact_indices),
                    )
                    return list(zip(fact_indices, [1.0] * len(fact_indices)))
                return ranked
        except Exception as e:
            log.warning("Reranker failed: %s", e)
            return list(zip(fact_indices, [1.0] * len(fact_indices)))

    # -- Personalized PageRank ---------------------------------------------

    def _ppr(
        self,
        seed_scores: dict[int, float],
        adj: dict[int, list[int]],
    ) -> dict[int, float]:
        """Personalized PageRank on the fact graph.

        Args:
            seed_scores: initial score for each seed fact (from reranker)
            adj: adjacency list (fact_idx → list of neighbor fact indices)

        Returns:
            dict mapping fact_idx → PPR score
        """
        if not seed_scores:
            return {}

        nodes = list(adj.keys())
        n = len(nodes)
        if n == 0:
            return {}

        # Initialize: distribute seed score uniformly among seeds
        teleport = np.zeros(n)
        node_to_local = {node: i for i, node in enumerate(nodes)}
        total_seed = sum(seed_scores.values())
        if total_seed == 0:
            return {}
        for fi, score in seed_scores.items():
            if fi in node_to_local:
                teleport[node_to_local[fi]] = score / total_seed

        # PPR: r = (1-d) * teleport + d * M^T * r
        # where M is the column-stochastic transition matrix
        scores = teleport.copy()
        for _ in range(self.ppr_iterations):
            new_scores = (1 - self.ppr_damping) * teleport
            for i, node in enumerate(nodes):
                neighbors = adj.get(node, [])
                if not neighbors:
                    # Dangling node: redistribute uniformly
                    new_scores += scores[i] / n
                    continue
                share = self.ppr_damping * scores[i] / len(neighbors)
                for nb in neighbors:
                    if nb in node_to_local:
                        new_scores[node_to_local[nb]] += share
            scores = new_scores

        result: dict[int, float] = {}
        for i, node in enumerate(nodes):
            if scores[i] > self.ppr_threshold:
                result[node] = float(scores[i])
        return result

    # -- passage scoring from PPR ------------------------------------------

    def _passages_from_ppr(
        self, ppr_scores: dict[int, float]
    ) -> list[tuple[int, float]]:
        """Aggregate passage scores from fact PPR scores.

        A passage's score = sum of PPR scores of its facts * passage_node_weight
        """
        passage_scores: dict[int, float] = {}
        for fi, score in ppr_scores.items():
            if fi not in self.memory.facts:
                continue
            fact = self.memory.facts[fi]
            for pi in fact.passage_indices:
                if pi in self.memory.passages:
                    passage_scores[pi] = passage_scores.get(pi, 0.0) + score * self.passage_node_weight

        ranked = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    # -- main retrieve -----------------------------------------------------

    async def retrieve(self, query: str) -> list[ChunkHit]:
        """Full MemGraphRAG retrieval pipeline.

        1. Score facts against query
        2. Rerank top facts
        3. Run PPR on fact graph
        4. Aggregate passage scores
        5. Return top-k passages as ChunkHits
        """
        log.info("MemGraphRAG retrieve: %r (memory: %s)", query[:80], self.memory.stats)

        # Step 1: Score facts
        all_scored = await self.score_facts(query)
        if not all_scored:
            log.info("No facts scored, returning empty")
            return []

        # Step 2: Take top facts for reranking
        top_fact_indices = [fi for fi, _ in all_scored[:self.fact_top_k]]
        top_fact_texts = [self.memory.facts[fi].triple_str for fi in top_fact_indices if fi in self.memory.facts]

        reranked = await self.rerank_facts(query, top_fact_indices, top_fact_texts)
        log.info("Reranked %d facts", len(reranked))

        # Step 3: PPR from reranked facts
        seed_scores = {fi: score for fi, score in reranked if score > 0}
        ppr_scores = self._ppr(seed_scores, self._fact_adj)
        log.info("PPR: %d facts above threshold", len(ppr_scores))

        # Step 4: Aggregate to passages
        passage_ranked = self._passages_from_ppr(ppr_scores)
        log.info("Passage candidates: %d", len(passage_ranked))

        # Step 5: Build ChunkHits
        hits: list[ChunkHit] = []
        for pi, score in passage_ranked[:self.top_k]:
            if pi in self.memory.passages:
                p = self.memory.passages[pi]
                hits.append(ChunkHit(
                    id=p.chunk_id,
                    text=p.content,
                    score=score,
                    source="memgraphrag",
                ))

        log.info("Returning %d hits", len(hits))
        return hits
