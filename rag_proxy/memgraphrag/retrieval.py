"""MemGraphRAG retrieval: fact scoring -> rerank -> PPR graph walk -> passage retrieval.

  1. Embed query, score facts via dense similarity
  2. Rerank the top facts with the cross-encoder sidecar when one is configured;
     otherwise the dense similarity scores stand
  3. Personalized PageRank over the fact graph, seeded by those scores
  4. Return top passages ranked by aggregated PPR score

Schema-layer filtering from the paper is not implemented; the schema layer only
supplies fact adjacency. When no facts score above zero similarity, returns an
empty list (no dense fallback).
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import numpy as np

from rag_proxy.clients.rerank_adapter import (
    parse_rerank_indices,
    rerank_request_payload,
    rerank_request_url,
)
from rag_proxy.config import settings
from rag_proxy.context import ChunkHit
from rag_proxy.memgraphrag.cache import MemoryIndex

try:
    from rag_proxy.sidecar_client import get_embed_client, get_reranker_client
except ImportError:
    get_embed_client = None  # type: ignore[assignment,misc]
    get_reranker_client = None  # type: ignore[assignment,misc]

log = logging.getLogger("rag-proxy.memgraphrag.retrieval")


class MemGraphRetriever:
    """Online retrieval using the three-layer memory graph."""

    def __init__(
        self,
        index: MemoryIndex,
        embed_url: str | None = None,
        reranker_url: str | None = None,
        top_k: int = 5,
        fact_top_k: int = 20,
        ppr_damping: float = 0.85,
        ppr_iterations: int = 20,
        ppr_threshold: float = 0.01,
        passage_node_weight: float = 0.5,
    ):
        self.index = index
        self.memory = index.memory
        self.embed_url = embed_url or settings.embed_url
        # Without ENABLE_RERANKER the sidecar is not running, so falling back to
        # settings.reranker_url would buy a timeout per query and nothing else.
        self.reranker_url = reranker_url or (
            settings.reranker_url if settings.enable_reranker else ""
        )
        self.top_k = top_k
        self.fact_top_k = fact_top_k
        self.ppr_damping = ppr_damping
        self.ppr_iterations = ppr_iterations
        self.ppr_threshold = ppr_threshold
        self.passage_node_weight = passage_node_weight

    # -- embedding ---------------------------------------------------------

    async def _embed_query(self, query: str) -> list[float]:
        """Embed a query string via the OpenAI-compatible embeddings API."""
        trimmed = query.strip()
        if not trimmed:
            return []
        url = f"{self.embed_url.rstrip('/')}/v1/embeddings"
        payload = {"model": settings.embed_model, "input": trimmed}
        if get_embed_client is not None:
            try:
                client = get_embed_client()
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()["data"][0]["embedding"]
            except RuntimeError:
                pass
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    # -- fact scoring ------------------------------------------------------

    def _score_facts_vectorized(self, query_emb: np.ndarray) -> list[tuple[int, float]]:
        """Score embedded facts via single matmul against row-normalized fact vectors."""
        if self.index.fact_embeddings.size == 0:
            return []

        query_norm = float(np.linalg.norm(query_emb))
        if query_norm == 0:
            return []

        query_unit = query_emb / query_norm
        scores = self.index.fact_embeddings @ query_unit

        order = np.argsort(-scores)
        return [
            (int(self.index.fact_indices[i]), float(scores[i]))
            for i in order
        ]

    async def score_facts(self, query: str) -> list[tuple[int, float]]:
        """Score all facts against the query using precomputed embedding cosine similarity.

        Query is embedded once via HTTP; fact vectors come from the cached index.
        Facts without stored embeddings are omitted at index build time.

        Returns list of (fact_idx, score) sorted by score descending.
        """
        query_emb = np.array(await self._embed_query(query), dtype=np.float32)
        return await asyncio.to_thread(self._score_facts_vectorized, query_emb)

    # -- reranking ---------------------------------------------------------

    async def rerank_facts(
        self, query: str, fact_indices: list[int], fact_texts: list[str]
    ) -> list[tuple[int, float]] | None:
        """Rerank facts using the cross-encoder sidecar or OpenAI-style /v1/rerank.

        Returns list of (fact_idx, score) sorted by score descending, or None
        when the sidecar is unavailable or unusable. None means "no opinion" so
        the caller keeps its own ordering; flattening to uniform scores here
        would seed PPR with every candidate fact weighted equally.
        """
        if not fact_indices or not self.reranker_url:
            return None

        pairs = [{"query": query, "document": text} for text in fact_texts]
        try:
            timeout = settings.rerank_timeout_ms / 1000.0 + 0.5
            url = rerank_request_url(self.reranker_url)
            payload = rerank_request_payload(pairs, len(fact_indices))
            resp = None
            if get_reranker_client is not None:
                try:
                    client = get_reranker_client()
                    resp = await client.post(url, json=payload, timeout=timeout)
                except RuntimeError:
                    resp = None
            if resp is None:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, json=payload)
            resp.raise_for_status()
            order = parse_rerank_indices(
                resp.json(),
                n_pairs=len(fact_indices),
                top_k=len(fact_indices),
            )
            if not order:
                return None
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
                return None
            return ranked
        except Exception as e:
            log.warning("Reranker failed: %s", e)
            return None

    # -- Personalized PageRank ---------------------------------------------

    def _ppr(
        self,
        seed_scores: dict[int, float],
        adj: dict[int, list[int]],
    ) -> dict[int, float]:
        """Personalized PageRank on the fact graph.

        Args:
            seed_scores: initial score for each seed fact (from reranker)
            adj: adjacency list (fact_idx -> list of neighbor fact indices)

        Returns:
            dict mapping fact_idx -> PPR score
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

        # Step 1: Score facts (embed async, score in thread)
        query_emb = np.array(await self._embed_query(query), dtype=np.float32)
        all_scored = await asyncio.to_thread(self._score_facts_vectorized, query_emb)
        if not all_scored:
            log.info("No facts scored, returning empty")
            return []

        # Step 2: Take top facts for reranking. Index and text lists must stay
        # aligned — the sidecar answers with positions into the pair list.
        top_scored = [
            (fi, score)
            for fi, score in all_scored[:self.fact_top_k]
            if fi in self.memory.facts
        ]
        top_fact_indices = [fi for fi, _ in top_scored]
        top_fact_texts = [self.memory.facts[fi].triple_str for fi in top_fact_indices]

        reranked = await self.rerank_facts(query, top_fact_indices, top_fact_texts)
        if reranked is None:
            # No reranker opinion: keep dense similarity as the PPR seed weights.
            reranked = top_scored
        else:
            log.info("Reranked %d facts", len(reranked))

        # Step 3: PPR from reranked facts
        seed_scores = {fi: score for fi, score in reranked if score > 0}
        ppr_scores = await asyncio.to_thread(
            self._ppr, seed_scores, self.index.fact_adj
        )
        log.info("PPR: %d facts above threshold", len(ppr_scores))

        # Step 4: Aggregate to passages
        passage_ranked = await asyncio.to_thread(self._passages_from_ppr, ppr_scores)
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
