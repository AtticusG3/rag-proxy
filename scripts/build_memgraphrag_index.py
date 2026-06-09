"""Build MemGraphRAG index from a text corpus.

Pipeline:
  1. Chunk text into passages
  2. Extract entities and relations (triples + schemas) using LLM
  3. Filter low-frequency ontologies (thematic denoising)
  4. Detect and resolve conflicts
  5. Build and save ThreeLayerMemory

Usage:
  python -m scripts.build_memgraphrag_index \
    --input corpus.txt \
    --output /var/lib/rag_proxy/memgraphrag.sqlite \
    --llm-url http://192.168.1.202:8080/v1 \
    --llm-model qwen3.5-9b-turbo \
    --chunk-size 512 \
    --overlap 64
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import hashlib
import os
import sys
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
        async with httpx.AsyncClient(timeout=120) as client:
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
- Output valid JSON only, no explanations"""

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
    """Extract entities from text using LLM."""
    try:
        raw = await llm.chat(ENTITY_PROMPT_SYSTEM, f"Text:\n{text}")
        # Find JSON in response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            return data.get("entities", [])
    except Exception as e:
        log.warning("Entity extraction failed: %s", e)
    return []

async def extract_relations(llm: LLMClient, text: str, entities: list[dict]) -> list[dict]:
    """Extract relations from text + entities using LLM."""
    try:
        entity_list = "\n".join(f"- {e['text']} ({e['type']})" for e in entities)
        user_msg = f"Text:\n{text}\n\nEntities:\n{entity_list}"
        raw = await llm.chat(RELATION_PROMPT_SYSTEM, user_msg)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            return data.get("triples", [])
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

def build_memory(chunks_data: list[dict], db_path: str) -> None:
    """Build ThreeLayerMemory from extracted chunks and save to SQLite."""
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rag_proxy.memgraphrag.memory import ThreeLayerMemory

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

    memory.save(db_path)
    log.info("Built memory: %s → %s", memory.stats, db_path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Build MemGraphRAG index")
    parser.add_argument("--input", required=True, help="Input text file or directory")
    parser.add_argument("--output", required=True, help="Output SQLite database path")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8080/v1", help="LLM API URL")
    parser.add_argument("--llm-model", default="qwen3.5-9b-turbo", help="LLM model name")
    parser.add_argument("--api-key", default="rag-proxy", help="LLM API key")
    parser.add_argument("--chunk-size", type=int, default=512, help="Tokens per chunk")
    parser.add_argument("--overlap", type=int, default=64, help="Token overlap between chunks")
    parser.add_argument("--min-schema-freq", type=int, default=2,
                        help="Minimum schema frequency (thematic denoising)")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Max chunks to process (0 = all)")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Max concurrent LLM requests")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Load input
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

    # Chunk
    all_chunks: list[dict] = []
    for text in texts:
        all_chunks.extend(chunk_text(text, args.chunk_size, args.overlap))
    if args.max_chunks > 0:
        all_chunks = all_chunks[:args.max_chunks]
    log.info("Created %d chunks", len(all_chunks))

    # Extract entities and relations
    llm = LLMClient(args.llm_url, args.llm_model, args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process_chunk(chunk: dict) -> dict:
        async with semaphore:
            entities = await extract_entities(llm, chunk["text"])
            chunk["entities"] = entities
            if entities:
                triples = await extract_relations(llm, chunk["text"], entities)
                chunk["triples"] = triples
            else:
                chunk["triples"] = []
            log.info("Chunk %s: %d entities, %d triples", chunk["chunk_id"], len(entities), len(chunk["triples"]))
            return chunk

    log.info("Extracting entities and relations (concurrency=%d)...", args.concurrency)
    chunks_data = await asyncio.gather(*[process_chunk(c) for c in all_chunks])

    total_entities = sum(len(c.get("entities", [])) for c in chunks_data)
    total_triples = sum(len(c.get("triples", [])) for c in chunks_data)
    log.info("Extracted %d entities, %d triples from %d chunks", total_entities, total_triples, len(chunks_data))

    # Filter low-frequency ontologies
    chunks_data = filter_ontologies(chunks_data, args.min_schema_freq)

    # Build memory
    build_memory(chunks_data, args.output)

    # Print stats
    total_triples_after = sum(len(c.get("triples", [])) for c in chunks_data)
    log.info("Done. %d triples retained after filtering. Output: %s", total_triples_after, args.output)


if __name__ == "__main__":
    asyncio.run(main())
