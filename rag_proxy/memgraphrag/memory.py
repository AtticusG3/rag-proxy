"""Three-layer memory: schema → fact → passage, with inter-layer indices.

SQLite-backed for persistence. Mirrors the MemGraphRAG paper's memory structure:
  Schema layer: ontology patterns (head_type, relation, tail_type) with frequency
  Fact layer:    triples (head, relation, tail) linked to schema + passages
  Passage layer: original text chunks linked to facts

Inter-layer edges:
  Schema --1:N--> Fact  (each fact belongs to exactly one schema)
  Fact    --N:M--> Passage (each fact extracted from N passages, each passage has M facts)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("rag-proxy.memgraphrag.memory")


# ---------------------------------------------------------------------------
# Schema layer
# ---------------------------------------------------------------------------

@dataclass
class SchemaNode:
    idx: int
    head_type: str
    relation: str
    tail_type: str
    frequency: int = 0
    fact_indices: list[int] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.head_type, self.relation, self.tail_type)


# ---------------------------------------------------------------------------
# Fact layer
# ---------------------------------------------------------------------------

@dataclass
class FactNode:
    idx: int
    head: str
    relation: str
    tail: str
    schema_idx: int = -1
    passage_indices: list[int] = field(default_factory=list)
    embedding: list[float] | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.head, self.relation, self.tail)

    @property
    def triple_str(self) -> str:
        return f"({self.head}, {self.relation}, {self.tail})"


# ---------------------------------------------------------------------------
# Passage layer
# ---------------------------------------------------------------------------

@dataclass
class PassageNode:
    idx: int
    chunk_id: str
    content: str
    fact_indices: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS schemas (
    idx         INTEGER PRIMARY KEY,
    head_type   TEXT NOT NULL,
    relation    TEXT NOT NULL,
    tail_type   TEXT NOT NULL,
    frequency   INTEGER DEFAULT 0,
    fact_indices TEXT DEFAULT '[]',
    UNIQUE(head_type, relation, tail_type)
);
"""

_FACT_DDL = """
CREATE TABLE IF NOT EXISTS facts (
    idx            INTEGER PRIMARY KEY,
    head           TEXT NOT NULL,
    relation       TEXT NOT NULL,
    tail           TEXT NOT NULL,
    schema_idx     INTEGER DEFAULT -1,
    passage_indices TEXT DEFAULT '[]',
    embedding      TEXT,
    UNIQUE(head, relation, tail)
);
"""

_PASSAGE_DDL = """
CREATE TABLE IF NOT EXISTS passages (
    idx         INTEGER PRIMARY KEY,
    chunk_id    TEXT NOT NULL,
    content     TEXT NOT NULL,
    fact_indices TEXT DEFAULT '[]',
    UNIQUE(chunk_id)
);
"""

# Inter-layer junction tables for fast traversal
_SCHEMA_FACT_DDL = """
CREATE TABLE IF NOT EXISTS schema_fact_edges (
    schema_idx  INTEGER NOT NULL,
    fact_idx    INTEGER NOT NULL,
    PRIMARY KEY (schema_idx, fact_idx),
    FOREIGN KEY (schema_idx) REFERENCES schemas(idx),
    FOREIGN KEY (fact_idx)   REFERENCES facts(idx)
);
"""

_FACT_PASSAGE_DDL = """
CREATE TABLE IF NOT EXISTS fact_passage_edges (
    fact_idx     INTEGER NOT NULL,
    passage_idx  INTEGER NOT NULL,
    PRIMARY KEY (fact_idx, passage_idx),
    FOREIGN KEY (fact_idx)    REFERENCES facts(idx),
    FOREIGN KEY (passage_idx) REFERENCES passages(idx)
);
"""


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(_SCHEMA_DDL)
    conn.execute(_FACT_DDL)
    conn.execute(_PASSAGE_DDL)
    conn.execute(_SCHEMA_FACT_DDL)
    conn.execute(_FACT_PASSAGE_DDL)
    fact_cols = {row[1] for row in conn.execute("PRAGMA table_info(facts)")}
    if "embedding" not in fact_cols:
        conn.execute("ALTER TABLE facts ADD COLUMN embedding TEXT")
    conn.commit()


# ---------------------------------------------------------------------------
# ThreeLayerMemory — in-memory + SQLite sync
# ---------------------------------------------------------------------------

class ThreeLayerMemory:
    """Three-layer memory with inter-layer connections."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else None
        self.schemas: dict[int, SchemaNode] = {}
        self.facts: dict[int, FactNode] = {}
        self.passages: dict[int, PassageNode] = {}
        self._schema_key_to_idx: dict[tuple[str, str, str], int] = {}
        self._fact_key_to_idx: dict[tuple[str, str, str], int] = {}
        self._chunk_id_to_idx: dict[str, int] = {}
        self._next_schema_idx = 0
        self._next_fact_idx = 0
        self._next_passage_idx = 0

        if self.db_path and self.db_path.exists():
            self._load_from_db()

    # -- schema -----------------------------------------------------------

    def add_schema(self, head_type: str, relation: str, tail_type: str) -> int:
        key = (head_type, relation, tail_type)
        if key in self._schema_key_to_idx:
            idx = self._schema_key_to_idx[key]
            self.schemas[idx].frequency += 1
            return idx
        idx = self._next_schema_idx
        self._next_schema_idx += 1
        node = SchemaNode(idx=idx, head_type=head_type, relation=relation, tail_type=tail_type, frequency=1)
        self.schemas[idx] = node
        self._schema_key_to_idx[key] = idx
        return idx

    # -- fact -------------------------------------------------------------

    def add_fact(self, head: str, relation: str, tail: str, schema_idx: int,
                 passage_idx: int) -> int:
        key = (head, relation, tail)
        if key in self._fact_key_to_idx:
            idx = self._fact_key_to_idx[key]
            node = self.facts[idx]
            if passage_idx not in node.passage_indices:
                node.passage_indices.append(passage_idx)
            return idx
        idx = self._next_fact_idx
        self._next_fact_idx += 1
        node = FactNode(
            idx=idx, head=head, relation=relation, tail=tail,
            schema_idx=schema_idx, passage_indices=[passage_idx],
        )
        self.facts[idx] = node
        self._fact_key_to_idx[key] = idx
        # link schema → fact
        if schema_idx in self.schemas:
            if idx not in self.schemas[schema_idx].fact_indices:
                self.schemas[schema_idx].fact_indices.append(idx)
        return idx

    def set_fact_embedding(self, fact_idx: int, embedding: list[float]) -> None:
        """Attach a precomputed embedding vector to a fact."""
        if fact_idx in self.facts:
            self.facts[fact_idx].embedding = embedding

    # -- passage ----------------------------------------------------------

    def add_passage(self, chunk_id: str, content: str, fact_indices: list[int]) -> int:
        if chunk_id in self._chunk_id_to_idx:
            idx = self._chunk_id_to_idx[chunk_id]
            node = self.passages[idx]
            for fi in fact_indices:
                if fi not in node.fact_indices:
                    node.fact_indices.append(fi)
            return idx
        idx = self._next_passage_idx
        self._next_passage_idx += 1
        node = PassageNode(idx=idx, chunk_id=chunk_id, content=content, fact_indices=list(fact_indices))
        self.passages[idx] = node
        self._chunk_id_to_idx[chunk_id] = idx
        return idx

    # -- inter-layer queries ----------------------------------------------

    def get_facts_for_passage(self, passage_idx: int) -> list[FactNode]:
        if passage_idx not in self.passages:
            return []
        return [self.facts[fi] for fi in self.passages[passage_idx].fact_indices if fi in self.facts]

    def get_passages_for_fact(self, fact_idx: int) -> list[PassageNode]:
        if fact_idx not in self.facts:
            return []
        return [self.passages[pi] for pi in self.facts[fact_idx].passage_indices if pi in self.passages]

    def get_schema_for_fact(self, fact_idx: int) -> SchemaNode | None:
        if fact_idx not in self.facts:
            return None
        si = self.facts[fact_idx].schema_idx
        return self.schemas.get(si)

    def get_facts_for_schema(self, schema_idx: int) -> list[FactNode]:
        if schema_idx not in self.schemas:
            return []
        return [self.facts[fi] for fi in self.schemas[schema_idx].fact_indices if fi in self.facts]

    # -- graph walk (for PPR) --------------------------------------------

    def get_related_fact_indices(self, fact_idx: int) -> list[int]:
        """Return fact indices that share a passage or schema with the given fact."""
        if fact_idx not in self.facts:
            return []
        related: set[int] = set()
        node = self.facts[fact_idx]
        # same schema
        if node.schema_idx in self.schemas:
            for fi in self.schemas[node.schema_idx].fact_indices:
                if fi != fact_idx:
                    related.add(fi)
        # same passages
        for pi in node.passage_indices:
            if pi in self.passages:
                for fi in self.passages[pi].fact_indices:
                    if fi != fact_idx:
                        related.add(fi)
        return list(related)

    # -- stats ------------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        return {
            "schemas": len(self.schemas),
            "facts": len(self.facts),
            "passages": len(self.passages),
        }

    # -- persistence ------------------------------------------------------

    def save(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else self.db_path
        if not path:
            raise ValueError("No db_path specified")
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        _ensure_tables(conn)
        conn.execute("DELETE FROM schema_fact_edges")
        conn.execute("DELETE FROM fact_passage_edges")
        conn.execute("DELETE FROM schemas")
        conn.execute("DELETE FROM facts")
        conn.execute("DELETE FROM passages")

        for s in self.schemas.values():
            conn.execute(
                "INSERT OR REPLACE INTO schemas (idx, head_type, relation, tail_type, frequency, fact_indices) VALUES (?,?,?,?,?,?)",
                (s.idx, s.head_type, s.relation, s.tail_type, s.frequency, json.dumps(s.fact_indices)),
            )
        for f in self.facts.values():
            emb_json = json.dumps(f.embedding) if f.embedding else None
            conn.execute(
                "INSERT OR REPLACE INTO facts (idx, head, relation, tail, schema_idx, passage_indices, embedding) VALUES (?,?,?,?,?,?,?)",
                (f.idx, f.head, f.relation, f.tail, f.schema_idx, json.dumps(f.passage_indices), emb_json),
            )
        for p in self.passages.values():
            conn.execute(
                "INSERT OR REPLACE INTO passages (idx, chunk_id, content, fact_indices) VALUES (?,?,?,?)",
                (p.idx, p.chunk_id, p.content, json.dumps(p.fact_indices)),
            )
        # junction tables
        for s in self.schemas.values():
            for fi in s.fact_indices:
                conn.execute("INSERT OR IGNORE INTO schema_fact_edges (schema_idx, fact_idx) VALUES (?,?)", (s.idx, fi))
        for f in self.facts.values():
            for pi in f.passage_indices:
                conn.execute("INSERT OR IGNORE INTO fact_passage_edges (fact_idx, passage_idx) VALUES (?,?)", (f.idx, pi))
        conn.commit()
        conn.close()
        log.info("Saved memory to %s: %s", path, self.stats)

    def _load_from_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        _ensure_tables(conn)
        for row in conn.execute("SELECT idx, head_type, relation, tail_type, frequency, fact_indices FROM schemas"):
            idx, ht, r, tt, freq, fi_json = row
            node = SchemaNode(idx=idx, head_type=ht, relation=r, tail_type=tt, frequency=freq,
                              fact_indices=json.loads(fi_json or "[]"))
            self.schemas[idx] = node
            self._schema_key_to_idx[(ht, r, tt)] = idx
            self._next_schema_idx = max(self._next_schema_idx, idx + 1)

        for row in conn.execute(
            "SELECT idx, head, relation, tail, schema_idx, passage_indices, embedding FROM facts"
        ):
            idx, h, r, t, si, pi_json, emb_json = row
            embedding = json.loads(emb_json) if emb_json else None
            node = FactNode(
                idx=idx, head=h, relation=r, tail=t, schema_idx=si,
                passage_indices=json.loads(pi_json or "[]"),
                embedding=embedding,
            )
            self.facts[idx] = node
            self._fact_key_to_idx[(h, r, t)] = idx
            self._next_fact_idx = max(self._next_fact_idx, idx + 1)

        for row in conn.execute("SELECT idx, chunk_id, content, fact_indices FROM passages"):
            idx, cid, content, fi_json = row
            node = PassageNode(idx=idx, chunk_id=cid, content=content,
                               fact_indices=json.loads(fi_json or "[]"))
            self.passages[idx] = node
            self._chunk_id_to_idx[cid] = idx
            self._next_passage_idx = max(self._next_passage_idx, idx + 1)

        conn.close()
        log.info("Loaded memory from %s: %s", self.db_path, self.stats)


def load_memory(db_path: str | Path) -> ThreeLayerMemory:
    """Load a ThreeLayerMemory from an SQLite database."""
    return ThreeLayerMemory(db_path=db_path)
