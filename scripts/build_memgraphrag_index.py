"""Build MemGraphRAG index from a text corpus OR a Qdrant collection.

Pipeline:
  1. Source:  --input corpus.txt | --source qdrant --qdrant-url ... --collection ...
  2. Sample chunks (proportional stratified by `source` field if from Qdrant)
  3. Extract entities and relations (triples + schemas) using LLM
  4. Filter low-frequency ontologies (thematic denoising)
  5. Build and save ThreeLayerMemory

Usage (file corpus, original mode):
  python -m scripts.build_memgraphrag_index \\
    --input corpus.txt \\
    --output /var/lib/rag_proxy/memgraphrag.sqlite \\
    --llm-url http://192.168.1.202:8080/v1 \\
    --llm-model qwen3.5-9b-turbo \\
    --chunk-size 512 \\
    --overlap 64

Usage (Qdrant collection, new mode):
  python -m scripts.build_memgraphrag_index \\
    --source qdrant \\
    --qdrant-url http://192.168.1.36:6333 \\
    --collection nomad_knowledge_base \\
    --output /var/lib/rag_proxy/memgraphrag.sqlite \\
    --llm-url http://192.168.1.202:8080/v1 \\
    --llm-model qwen3.5-9b-turbo \\
    --max-chunks 1000 \\
    --stratify-field source
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import hashlib
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("memgraphrag.build")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """Simple token-based chunking (whitespace tokens, no tokenizer dependency)."""
    tokens = text.split()
    chunks: list[dict] = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size
    i = 0
    idx = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = " ".join(chunk_tokens)
        cid = hashlib.md5(chunk_text.encode()).hexdigest()[:16]
        chunks.append({
            "chunk_id": f"chunk_{idx}_{cid}",
            "text": chunk_text,
            "token_count": len(chunk_tokens),
        })
        i += step
        idx += 1
    return chunks

# ---------------------------------------------------------------------------
# Qdrant source: stratified sample of pre-chunked payloads
# ---------------------------------------------------------------------------

async def fetch_qdrant_chunks(
    qdrant_url: str,
    collection: str,
    target_count: int,
    stratify_field: str = "source",
    payload_text_field: str = "text",
    seed: int = 42,
) -> list[dict]:
    """Fetch a stratified random sample of chunks from a Qdrant collection.

    Sampling is proportional: each distinct value of `stratify_field` gets
    ceil(target_count * count_in_field / total_in_collection) chunks.

    Each returned chunk has:
        chunk_id  : Qdrant point UUID (string)
        text      : payload[payload_text_field]
        source    : payload[stratify_field] (for traceability)
        meta      : {k: v for k, v in payload.items() if k != payload_text_field}
    """
    qdrant_url = qdrant_url.rstrip("/")
    rng = random.Random(seed)

    # Step 1: Get the field value distribution via facet
    log.info("Fetching %s distribution for stratified sampling...", stratify_field)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{qdrant_url}/collections/{collection}/facet",
            json={"key": stratify_field, "limit": 200},
        )
        resp.raise_for_status()
        facet = resp.json()["result"]["hits"]
    log.info("Found %d distinct %s values", len(facet), stratify_field)

    # Step 2: Compute sample size per value
    total = sum(h["count"] for h in facet)
    sample_plan: list[tuple[str, int]] = []
    remaining = target_count

    if target_count < len(facet):
        # For small sample sizes, pick top N sources by count and take 1 each
        # (proportional allocation can't give 1 chunk to 100 sources when target=10)
        log.info("Target %d < %d sources: using top-source strategy (1 chunk per top source)",
                 target_count, len(facet))
        for hit in facet[:target_count]:
            sample_plan.append((hit["value"], 1))
    else:
        # Proportional allocation with minimum of 1
        for hit in facet:
            value = hit["value"]
            allocated = max(1, round(target_count * hit["count"] / total))
            allocated = min(allocated, remaining)
            sample_plan.append((value, allocated))
            remaining -= allocated
            if remaining <= 0:
                break
    log.info("Sample plan: %d values, %d total chunks targeted", len(sample_plan), target_count)

    # Step 3: For each value, fetch N random chunks using offset_pagination scroll
    all_chunks: list[dict] = []
    async with httpx.AsyncClient(timeout=120) as client:
        for value, n in sample_plan:
            offset = None
            value_chunks: list[dict] = []
            # Scroll with filter on this value
            # We use a filter to get only points with this value, then random-offset
            attempts = 0
            while len(value_chunks) < n and attempts < n * 4 + 5:
                attempts += 1
                filter_ = {
                    "must": [{"key": stratify_field, "match": {"value": value}}]
                }
                params: dict[str, Any] = {
                    "limit": min(50, n - len(value_chunks)),
                    "with_payload": True,
                    "with_vectors": False,
                    "filter": filter_,
                }
                if offset is not None:
                    params["offset"] = offset
                resp = await client.post(
                    f"{qdrant_url}/collections/{collection}/points/scroll",
                    json=params,
                )
                resp.raise_for_status()
                result = resp.json()["result"]
                points = result.get("points", [])
                if not points:
                    break  # exhausted this value
                for pt in points:
                    payload = pt.get("payload", {})
                    text = payload.get(payload_text_field)
                    if not text:
                        continue
                    meta = {k: v for k, v in payload.items() if k != payload_text_field}
                    value_chunks.append({
                        "chunk_id": str(pt["id"]),
                        "text": text,
                        "source": value,
                        "meta": meta,
                    })
                offset = result.get("next_page_offset")
                if offset is None:
                    break  # no more pages
            # Shuffle and take exactly n
            rng.shuffle(value_chunks)
            all_chunks.extend(value_chunks[:n])
            log.info("  %s: %d/%d chunks fetched", value[-60:], len(value_chunks[:n]), n)

    log.info("Fetched %d chunks total (target was %d)", len(all_chunks), target_count)
    return all_chunks

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """Minimal OpenAI-compatible LLM client."""

    def __init__(self, base_url: str, model: str, api_key: str = "rag-proxy", temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self._cache: dict[str, str] = {}

    async def chat(self, system: str, user: str, max_tokens: int = 1024) -> str:
        key = hashlib.md5(f"{system}{user}".encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            self._cache[key] = text
            return text

# ---------------------------------------------------------------------------
# Extraction prompts
# ---------------------------------------------------------------------------

ENTITY_PROMPT_SYSTEM = """You are an expert information extraction system.
Extract ALL named entities and their types from the text.

Output JSON only:
{"entities": [{"text": "entity name", "type": "PERSON|ORGANIZATION|LOCATION|EVENT|CONCEPT|OBJECT|DATE|OTHER"}]}

Rules:
- Include persons, organizations, locations, events, concepts, objects, dates
- Use the most specific type possible
- Do NOT invent entities not in the text
- Output valid JSON only, no explanations

IMPORTANT: Respond with JSON only. No preamble, no explanation, no markdown code blocks."""

RELATION_PROMPT_SYSTEM = """You are an expert relation extraction system.
Given text and extracted entities, identify all meaningful relationships.

Output JSON only:
{
  "triples": [
    {
      "head": "entity text",
      "head_type": "PERSON|ORGANIZATION|...",
      "relation": "relationship verb/phrase",
      "tail": "entity text",
      "tail_type": "PERSON|ORGANIZATION|...",
      "schema": ["head_type", "relation", "tail_type"]
    }
  ]
}

Rules:
- Only relate entities that appear in the entity list
- Relations must be grounded in the text
- Use concise, specific relation phrases (e.g., "founded", "located_in", "participated_in")
- Output valid JSON only, no explanations"""

async def extract_entities(llm: LLMClient, text: str) -> list[dict]:
    """Extract entities from text using LLM. Returns [] on failure (with warning)."""
    try:
        raw = await llm.chat(ENTITY_PROMPT_SYSTEM, f"Text:\n{text}")
        # Find JSON in response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            return data.get("entities", [])
        log.warning("Entity extraction: no JSON object found in response (len=%d)", len(raw))
    except Exception as e:
        log.warning("Entity extraction failed: %s", e)
    return []

async def extract_relations(llm: LLMClient, text: str, entities: list[dict]) -> list[dict]:
    """Extract relations from text + entities using LLM. Returns [] on failure (with warning)."""
    try:
        entity_list = "\n".join(f"- {e['text']} ({e['type']})" for e in entities)
        user_msg = f"Text:\n{text}\n\nEntities:\n{entity_list}"
        raw = await llm.chat(RELATION_PROMPT_SYSTEM, user_msg)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            return data.get("triples", [])
        log.warning("Relation extraction: no JSON object found in response (len=%d)", len(raw))
    except Exception as e:
        log.warning("Relation extraction failed: %s", e)
    return []

# ---------------------------------------------------------------------------
# Ontology filtering (thematic denoising)
# ---------------------------------------------------------------------------

def filter_ontologies(
    chunks_data: list[dict],
    min_frequency: int = 2,
) -> list[dict]:
    """Remove triples whose schema appears fewer than min_frequency times."""
    # Count schema frequencies
    schema_counts: Counter = Counter()
    for chunk in chunks_data:
        for triple in chunk.get("triples", []):
            schema = triple.get("schema", [])
            if len(schema) == 3:
                schema_counts[tuple(schema)] += 1

    # Filter
    filtered = 0
    for chunk in chunks_data:
        original = chunk.get("triples", [])
        chunk["triples"] = [
            t for t in original
            if len(t.get("schema", [])) == 3 and schema_counts.get(tuple(t["schema"]), 0) >= min_frequency
        ]
        filtered += len(original) - len(chunk["triples"])

    log.info("Filtered %d low-frequency triples (min_freq=%d)", filtered, min_frequency)
    return chunks_data

# ---------------------------------------------------------------------------
# Build memory
# ---------------------------------------------------------------------------

def embed_fact_embeddings(memory: Any, embed_url: str, batch_size: int = 32) -> None:
    """Batch-embed all fact triples and attach vectors to memory nodes."""
    from ingest.embedder import embed_texts

    fact_items = list(memory.facts.items())
    if not fact_items:
        return
    log.info("Embedding %d facts (batch_size=%d)...", len(fact_items), batch_size)
    for i in range(0, len(fact_items), batch_size):
        batch = fact_items[i:i + batch_size]
        texts = [fact.triple_str for _, fact in batch]
        embeddings = embed_texts(texts, embed_url=embed_url)
        for (fi, _), emb in zip(batch, embeddings):
            memory.set_fact_embedding(fi, emb)
    log.info("Embedded %d facts", len(fact_items))


def build_memory(chunks_data: list[dict], db_path: str, embed_url: str | None = None) -> None:
    """Build ThreeLayerMemory from extracted chunks and save to SQLite."""
    # Import here to avoid circular imports / avoid loading retrieval (which needs rag_proxy.config)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rag_proxy.memgraphrag.memory import ThreeLayerMemory  # direct import, bypass __init__.py

    memory = ThreeLayerMemory()

    for chunk in chunks_data:
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        triples = chunk.get("triples", [])

        fact_indices = []
        for triple in triples:
            head = triple.get("head", "")
            relation = triple.get("relation", "")
            tail = triple.get("tail", "")
            schema = triple.get("schema", [])
            if not head or not relation or not tail or len(schema) != 3:
                continue

            schema_idx = memory.add_schema(schema[0], schema[1], schema[2])
            # Passage will be added below; for now use placeholder
            fact_idx = memory.add_fact(head, relation, tail, schema_idx, passage_idx=0)
            fact_indices.append(fact_idx)

        # Add passage with fact indices (update fact → passage links)
        passage_idx = memory.add_passage(chunk_id, text, fact_indices)

        # Update fact nodes to point back to this passage
        for fi in fact_indices:
            if fi in memory.facts and passage_idx not in memory.facts[fi].passage_indices:
                memory.facts[fi].passage_indices.append(passage_idx)

    if embed_url:
        embed_fact_embeddings(memory, embed_url)

    memory.save(db_path)
    log.info("Built memory: %s → %s", memory.stats, db_path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Build MemGraphRAG index")
    # Input source: either --input (file/dir) or --source qdrant
    parser.add_argument("--input", help="Input text file or directory of .txt files")
    parser.add_argument("--source", choices=["file", "qdrant"], default="file",
                        help="Input source type: 'file' (use --input) or 'qdrant' (use --qdrant-url/--collection)")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333",
                        help="Qdrant base URL (when --source qdrant)")
    parser.add_argument("--collection", default="nomad_knowledge_base",
                        help="Qdrant collection name (when --source qdrant)")
    parser.add_argument("--stratify-field", default="source",
                        help="Payload field to stratify sampling by (default: source)")
    parser.add_argument("--payload-text-field", default="text",
                        help="Payload field containing the chunk text (default: text)")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Random seed for stratified sampling (default: 42)")

    parser.add_argument("--output", required=True, help="Output SQLite database path")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8080/v1",
                        help="LLM API URL (OpenAI-compatible, e.g. llama-swap)")
    parser.add_argument("--llm-model", default="qwen3.5-9b-turbo",
                        help="LLM model name (alias for Qwen3.5-9B-Abliterated-Claude-4.6-Opus-Reasoning-Distilled)")
    parser.add_argument("--api-key", default="sk-llama-cpp", help="LLM API key")
    parser.add_argument("--chunk-size", type=int, default=512, help="Tokens per chunk (file mode only)")
    parser.add_argument("--overlap", type=int, default=64, help="Token overlap between chunks (file mode only)")
    parser.add_argument("--min-schema-freq", type=int, default=2,
                        help="Minimum schema frequency (thematic denoising)")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Max chunks to process (0 = all; for Qdrant this is the stratified sample size)")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Max concurrent LLM requests")
    parser.add_argument("--max-chars", type=int, default=2000,
                        help="Max characters of chunk text to send to LLM (truncate longer chunks)")
    parser.add_argument("--skip-relations", action="store_true",
                        help="Skip relation extraction (entities only — 50%% faster)")
    parser.add_argument("--embed-url", default=os.getenv("EMBED_URL", "http://127.0.0.1:8089"),
                        help="Embedding API URL for fact vectors (default: EMBED_URL or 127.0.0.1:8089)")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip fact embedding at build time (online scoring will skip those facts)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # ---- Load chunks ----
    t0 = time.time()
    if args.source == "qdrant":
        # Backward compat: if --input not given, treat --max-chunks as the sample size
        target = args.max_chunks if args.max_chunks > 0 else 1000
        log.info("Loading %d stratified chunks from Qdrant %s/%s", target, args.qdrant_url, args.collection)
        all_chunks = await fetch_qdrant_chunks(
            qdrant_url=args.qdrant_url,
            collection=args.collection,
            target_count=target,
            stratify_field=args.stratify_field,
            payload_text_field=args.payload_text_field,
            seed=args.sample_seed,
        )
    else:
        # File/dir mode (original)
        if not args.input:
            log.error("--input is required when --source=file")
            sys.exit(1)
        input_path = Path(args.input)
        if input_path.is_file():
            texts = [input_path.read_text(encoding="utf-8")]
        elif input_path.is_dir():
            texts = []
            for f in sorted(input_path.glob("**/*.txt")):
                texts.append(f.read_text(encoding="utf-8"))
            log.info("Loaded %d text files from %s", len(texts), input_path)
        else:
            log.error("Input path not found: %s", input_path)
            sys.exit(1)

        all_chunks = []
        for text in texts:
            all_chunks.extend(chunk_text(text, args.chunk_size, args.overlap))
        if args.max_chunks > 0:
            all_chunks = all_chunks[:args.max_chunks]
        log.info("Created %d chunks from file corpus", len(all_chunks))

    log.info("Chunk loading took %.1fs. %d chunks ready for LLM extraction.", time.time() - t0, len(all_chunks))
    if not all_chunks:
        log.error("No chunks to process. Aborting.")
        sys.exit(1)

    # ---- LLM extraction with throughput tracking ----
    llm = LLMClient(args.llm_url, args.llm_model, args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    json_failures = {"entity": 0, "relation": 0}
    extract_t0 = time.time()

    async def process_chunk(idx: int, chunk: dict) -> dict:
        async with semaphore:
            t_chunk = time.time()
            # Truncate chunk text to bound LLM input cost
            text = chunk["text"]
            if len(text) > args.max_chars:
                text = text[:args.max_chars] + "..."
                chunk["text_truncated"] = True
            else:
                chunk["text_truncated"] = False

            entities = await extract_entities(llm, text)
            if not entities:
                json_failures["entity"] += 1
            if entities and not args.skip_relations:
                triples = await extract_relations(llm, text, entities)
                if not triples:
                    json_failures["relation"] += 1
                chunk["triples"] = triples
            else:
                chunk["triples"] = []

            chunk["entities"] = entities
            elapsed = time.time() - t_chunk
            log.info("[%d/%d] %s: %d entities, %d triples (%.1fs)",
                     idx + 1, len(all_chunks), chunk["chunk_id"][:30],
                     len(entities), len(chunk["triples"]), elapsed)
            return chunk

    log.info("Extracting entities and relations (concurrency=%d)...", args.concurrency)
    chunks_data = await asyncio.gather(*[process_chunk(i, c) for i, c in enumerate(all_chunks)])

    extract_elapsed = time.time() - extract_t0
    total_entities = sum(len(c.get("entities", [])) for c in chunks_data)
    total_triples = sum(len(c.get("triples", [])) for c in chunks_data)
    throughput = len(all_chunks) / extract_elapsed if extract_elapsed > 0 else 0
    log.info(
        "Extraction complete: %d entities, %d triples from %d chunks in %.1fs (%.2f chunks/sec)",
        total_entities, total_triples, len(chunks_data), extract_elapsed, throughput,
    )
    log.info(
        "JSON parse failures: entity=%d, relation=%d (out of %d chunks each)",
        json_failures["entity"], json_failures["relation"], len(all_chunks),
    )

    # ---- Filter low-frequency ontologies ----
    chunks_data = filter_ontologies(chunks_data, args.min_schema_freq)

    # ---- Build memory ----
    build_t0 = time.time()
    embed_url = None if args.skip_embed else args.embed_url
    build_memory(chunks_data, args.output, embed_url=embed_url)
    build_elapsed = time.time() - build_t0
    log.info("Memory build took %.1fs. Output: %s", build_elapsed, args.output)

    # ---- Final stats ----
    total_triples_after = sum(len(c.get("triples", [])) for c in chunks_data)
    log.info(
        "DONE. %d chunks → %d entities → %d triples (filtered to %d). Total wall time: %.1fs. Output: %s",
        len(chunks_data), total_entities, total_triples, total_triples_after,
        time.time() - t0, args.output,
    )


if __name__ == "__main__":
    asyncio.run(main())
