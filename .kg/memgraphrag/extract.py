#!/usr/bin/env python3
"""Extract MemGraphRAG ontology meta-model and optional SQLite corpus instances."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

KG = Path(__file__).resolve().parent
REPO = KG.parent.parent

ENTITY_CLASSES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "EVENT",
    "CONCEPT",
    "OBJECT",
    "DATE",
    "OTHER",
]

LAYERS: list[tuple[str, str, str]] = [
    ("schema", "Ontology patterns (head_type, relation, tail_type)", "schemas"),
    ("fact", "Grounded entity-relation triples", "facts"),
    ("passage", "Source text chunks", "passages"),
]

COMPONENTS: list[tuple[str, str, str]] = [
    ("ThreeLayerMemory", "rag_proxy/memgraphrag/memory.py", "SQLite-backed three-layer store"),
    ("MemGraphRetriever", "rag_proxy/memgraphrag/retrieval.py", "Online PPR retrieval"),
    ("build_memgraphrag_index", "scripts/build_memgraphrag_index.py", "Offline LLM extraction indexer"),
    ("tier3_memgraphrag", "rag_proxy/stages/tier3_memgraphrag.py", "Cognitive pipeline stage"),
]

RETRIEVAL_STEPS = [
    "score_facts",
    "rerank_facts",
    "personalized_pagerank",
    "passage_aggregate",
]

# Canonical test fixture from tests/test_memgraphrag_retrieval.py
SAMPLE_ONTOLOGY = {
    "pattern": ("Person", "knows", "Person"),
    "fact": ("Alice", "knows", "Bob"),
    "passage": ("chunk-1", "Alice knows Bob in the lab."),
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def pattern_id(head_type: str, relation: str, tail_type: str) -> str:
    return f"mgr-pattern:{slug(head_type)}-{slug(relation)}-{slug(tail_type)}"


def entity_id(text: str) -> str:
    return f"mgr-entity:{slug(text)}"


def fact_id(head: str, relation: str, tail: str) -> str:
    return f"mgr-fact:{slug(head)}-{slug(relation)}-{slug(tail)}"


def passage_id(chunk_id: str) -> str:
    return f"mgr-passage:{slug(chunk_id)}"


def class_id(name: str) -> str:
    return f"mgr-class:{slug(name)}"


def ensure_entity_class(
    nodes: list[dict],
    edges: list[dict],
    root_id: str,
    class_name: str,
    source: str,
    extracted_at: str,
) -> str:
    """Create entity class node if missing (patterns may use Person vs PERSON)."""
    cid = class_id(class_name)
    if not any(n["id"] == cid for n in nodes):
        nodes.append(
            {
                "id": cid,
                "type": "MgrEntityClass",
                "name": class_name,
                "source": source,
                "extracted_at": extracted_at,
            }
        )
        edges.append(
            {
                "from": root_id,
                "to": cid,
                "type": "ALLOWS_CLASS",
                "source": source,
                "extracted_at": extracted_at,
                "confidence": 1.0,
            }
        )
    return cid


def export_sqlite(db_path: Path, extracted_at: str) -> tuple[list[dict], list[dict]]:
    """Read schemas, facts, passages from memgraphrag.sqlite."""
    nodes: list[dict] = []
    edges: list[dict] = []
    if not db_path.exists():
        return nodes, edges

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    for row in conn.execute(
        "SELECT head_type, relation, tail_type, frequency FROM schemas ORDER BY frequency DESC"
    ):
        pid = pattern_id(row["head_type"], row["relation"], row["tail_type"])
        nodes.append(
            {
                "id": pid,
                "type": "MgrOntologyPattern",
                "head_type": row["head_type"],
                "relation": row["relation"],
                "tail_type": row["tail_type"],
                "frequency": row["frequency"],
                "source": str(db_path),
                "extracted_at": extracted_at,
            }
        )
        head_cid = ensure_entity_class(
            nodes, edges, "mgr-root:memgraphrag", row["head_type"], str(db_path), extracted_at
        )
        tail_cid = ensure_entity_class(
            nodes, edges, "mgr-root:memgraphrag", row["tail_type"], str(db_path), extracted_at
        )
        edges.append(
            {
                "from": pid,
                "to": head_cid,
                "type": "HAS_HEAD_CLASS",
                "source": str(db_path),
                "extracted_at": extracted_at,
                "confidence": 1.0,
            }
        )
        edges.append(
            {
                "from": pid,
                "to": tail_cid,
                "type": "HAS_TAIL_CLASS",
                "source": str(db_path),
                "extracted_at": extracted_at,
                "confidence": 1.0,
            }
        )
        edges.append(
            {
                "from": "mgr-root:memgraphrag",
                "to": pid,
                "type": "CONTAINS",
                "source": str(db_path),
                "extracted_at": extracted_at,
                "confidence": 1.0,
            }
        )

    for row in conn.execute("SELECT head, relation, tail, schema_idx FROM facts"):
        head, relation, tail = row["head"], row["relation"], row["tail"]
        fid = fact_id(head, relation, tail)
        hid, tid = entity_id(head), entity_id(tail)
        for ent, eid in ((head, hid), (tail, tid)):
            nodes.append(
                {
                    "id": eid,
                    "type": "MgrEntity",
                    "name": ent,
                    "source": str(db_path),
                    "extracted_at": extracted_at,
                }
            )
        nodes.append(
            {
                "id": fid,
                "type": "MgrFact",
                "head": head,
                "relation": relation,
                "tail": tail,
                "source": str(db_path),
                "extracted_at": extracted_at,
            }
        )
        edges.extend(
            [
                {
                    "from": fid,
                    "to": hid,
                    "type": "HAS_HEAD",
                    "source": str(db_path),
                    "extracted_at": extracted_at,
                    "confidence": 1.0,
                },
                {
                    "from": fid,
                    "to": tid,
                    "type": "HAS_TAIL",
                    "source": str(db_path),
                    "extracted_at": extracted_at,
                    "confidence": 1.0,
                },
            ]
        )
        schema_row = conn.execute(
            "SELECT head_type, relation, tail_type FROM schemas WHERE idx = ?",
            (row["schema_idx"],),
        ).fetchone()
        if schema_row:
            pid = pattern_id(schema_row["head_type"], schema_row["relation"], schema_row["tail_type"])
            edges.append(
                {
                    "from": fid,
                    "to": pid,
                    "type": "CONFORMS_TO",
                    "source": str(db_path),
                    "extracted_at": extracted_at,
                    "confidence": 1.0,
                }
            )

    for row in conn.execute("SELECT chunk_id, content FROM passages"):
        pid = passage_id(row["chunk_id"])
        content = row["content"] or ""
        nodes.append(
            {
                "id": pid,
                "type": "MgrPassage",
                "chunk_id": row["chunk_id"],
                "content_preview": content[:200],
                "source": str(db_path),
                "extracted_at": extracted_at,
            }
        )
        edges.append(
            {
                "from": "mgr-root:memgraphrag",
                "to": pid,
                "type": "CONTAINS",
                "source": str(db_path),
                "extracted_at": extracted_at,
                "confidence": 1.0,
            }
        )

    for row in conn.execute("SELECT fact_idx, passage_idx FROM fact_passage_edges"):
        fact_row = conn.execute(
            "SELECT head, relation, tail FROM facts WHERE idx = ?", (row["fact_idx"],)
        ).fetchone()
        passage_row = conn.execute(
            "SELECT chunk_id FROM passages WHERE idx = ?", (row["passage_idx"],)
        ).fetchone()
        if fact_row and passage_row:
            fid = fact_id(fact_row["head"], fact_row["relation"], fact_row["tail"])
            pid = passage_id(passage_row["chunk_id"])
            edges.append(
                {
                    "from": fid,
                    "to": pid,
                    "type": "EXTRACTED_IN",
                    "source": str(db_path),
                    "extracted_at": extracted_at,
                    "confidence": 1.0,
                }
            )

    conn.close()
    return nodes, edges


def add_sample_fixture(
    nodes: list[dict],
    edges: list[dict],
    extracted_at: str,
) -> None:
    """Add deterministic sample ontology from unit-test fixture."""
    head_type, relation, tail_type = SAMPLE_ONTOLOGY["pattern"]
    head, rel, tail = SAMPLE_ONTOLOGY["fact"]
    chunk_id, content = SAMPLE_ONTOLOGY["passage"]

    pid = pattern_id(head_type, relation, tail_type)
    nodes.append(
        {
            "id": pid,
            "type": "MgrOntologyPattern",
            "head_type": head_type,
            "relation": relation,
            "tail_type": tail_type,
            "frequency": 1,
            "source": "tests/test_memgraphrag_retrieval.py",
            "extracted_at": extracted_at,
        }
    )
    for cls in (head_type, tail_type):
        cid = ensure_entity_class(
            nodes, edges, "mgr-root:memgraphrag", cls, "tests/test_memgraphrag_retrieval.py", extracted_at
        )
        edges.append(
            {
                "from": pid,
                "to": cid,
                "type": "HAS_HEAD_CLASS" if cls == head_type else "HAS_TAIL_CLASS",
                "source": "tests/test_memgraphrag_retrieval.py",
                "extracted_at": extracted_at,
                "confidence": 1.0,
            }
        )

    fid = fact_id(head, rel, tail)
    hid, tid = entity_id(head), entity_id(tail)
    for name, eid in ((head, hid), (tail, tid)):
        nodes.append(
            {
                "id": eid,
                "type": "MgrEntity",
                "name": name,
                "source": "tests/test_memgraphrag_retrieval.py",
                "extracted_at": extracted_at,
            }
        )
    nodes.append(
        {
            "id": fid,
            "type": "MgrFact",
            "head": head,
            "relation": rel,
            "tail": tail,
            "source": "tests/test_memgraphrag_retrieval.py",
            "extracted_at": extracted_at,
        }
    )
    passage = passage_id(chunk_id)
    nodes.append(
        {
            "id": passage,
            "type": "MgrPassage",
            "chunk_id": chunk_id,
            "content_preview": content[:200],
            "source": "tests/test_memgraphrag_retrieval.py",
            "extracted_at": extracted_at,
        }
    )
    edges.extend(
        [
            {
                "from": fid,
                "to": hid,
                "type": "HAS_HEAD",
                "source": "tests/test_memgraphrag_retrieval.py",
                "extracted_at": extracted_at,
                "confidence": 1.0,
            },
            {
                "from": fid,
                "to": tid,
                "type": "HAS_TAIL",
                "source": "tests/test_memgraphrag_retrieval.py",
                "extracted_at": extracted_at,
                "confidence": 1.0,
            },
            {
                "from": fid,
                "to": pid,
                "type": "CONFORMS_TO",
                "source": "tests/test_memgraphrag_retrieval.py",
                "extracted_at": extracted_at,
                "confidence": 1.0,
            },
            {
                "from": fid,
                "to": passage,
                "type": "EXTRACTED_IN",
                "source": "tests/test_memgraphrag_retrieval.py",
                "extracted_at": extracted_at,
                "confidence": 1.0,
            },
            {
                "from": "mgr-root:memgraphrag",
                "to": pid,
                "type": "CONTAINS",
                "source": "tests/test_memgraphrag_retrieval.py",
                "extracted_at": extracted_at,
                "confidence": 1.0,
            },
            {
                "from": "mgr-root:memgraphrag",
                "to": passage,
                "type": "CONTAINS",
                "source": "tests/test_memgraphrag_retrieval.py",
                "extracted_at": extracted_at,
                "confidence": 1.0,
            },
        ]
    )


def build_meta_model(extracted_at: str) -> tuple[list[dict], list[dict]]:
    nodes: list[dict] = []
    edges: list[dict] = []

    nodes.append(
        {
            "id": "mgr-root:memgraphrag",
            "type": "MgrRoot",
            "name": "memgraphrag",
            "source": "rag_proxy/memgraphrag/memory.py",
            "extracted_at": extracted_at,
        }
    )

    for layer_key, description, table in LAYERS:
        lid = f"mgr-layer:{layer_key}"
        nodes.append(
            {
                "id": lid,
                "type": "MgrLayer",
                "name": layer_key,
                "table": table,
                "description": description,
                "source": "rag_proxy/memgraphrag/memory.py",
                "extracted_at": extracted_at,
            }
        )
        edges.append(
            {
                "from": "mgr-root:memgraphrag",
                "to": lid,
                "type": "CONTAINS",
                "source": "rag_proxy/memgraphrag/memory.py",
                "extracted_at": extracted_at,
                "confidence": 1.0,
            }
        )

    for cls in ENTITY_CLASSES:
        cid = class_id(cls)
        nodes.append(
            {
                "id": cid,
                "type": "MgrEntityClass",
                "name": cls,
                "source": "scripts/build_memgraphrag_index.py:ENTITY_PROMPT_SYSTEM",
                "extracted_at": extracted_at,
            }
        )
        edges.append(
            {
                "from": "mgr-root:memgraphrag",
                "to": cid,
                "type": "ALLOWS_CLASS",
                "source": "scripts/build_memgraphrag_index.py",
                "extracted_at": extracted_at,
                "confidence": 1.0,
            }
        )

    comp_ids: dict[str, str] = {}
    for name, path, role in COMPONENTS:
        comp_id = f"mgr-component:{slug(name)}"
        comp_ids[name] = comp_id
        nodes.append(
            {
                "id": comp_id,
                "type": "MgrComponent",
                "name": name,
                "path": path,
                "role": role,
                "source": path,
                "extracted_at": extracted_at,
            }
        )
        edges.append(
            {
                "from": "mgr-root:memgraphrag",
                "to": comp_id,
                "type": "CONTAINS",
                "source": path,
                "extracted_at": extracted_at,
                "confidence": 1.0,
            }
        )

    memory_id = comp_ids["ThreeLayerMemory"]
    for layer_key, _, _ in LAYERS:
        edges.append(
            {
                "from": comp_ids["ThreeLayerMemory"],
                "to": f"mgr-layer:{layer_key}",
                "type": "IMPLEMENTS",
                "source": "rag_proxy/memgraphrag/memory.py",
                "extracted_at": extracted_at,
                "confidence": 1.0,
            }
        )
    edges.append(
        {
            "from": comp_ids["MemGraphRetriever"],
            "to": memory_id,
            "type": "USES",
            "source": "rag_proxy/memgraphrag/retrieval.py",
            "extracted_at": extracted_at,
            "confidence": 1.0,
        }
    )
    edges.append(
        {
            "from": comp_ids["tier3_memgraphrag"],
            "to": comp_ids["MemGraphRetriever"],
            "type": "USES",
            "source": "rag_proxy/stages/tier3_memgraphrag.py",
            "extracted_at": extracted_at,
            "confidence": 1.0,
        }
    )
    edges.append(
        {
            "from": comp_ids["build_memgraphrag_index"],
            "to": "mgr-layer:schema",
            "type": "CONTAINS",
            "source": "scripts/build_memgraphrag_index.py",
            "extracted_at": extracted_at,
            "confidence": 1.0,
        }
    )

    step_ids = [f"mgr-step:{slug(s)}" for s in RETRIEVAL_STEPS]
    for step, sid in zip(RETRIEVAL_STEPS, step_ids):
        nodes.append(
            {
                "id": sid,
                "type": "MgrComponent",
                "name": step,
                "role": "retrieval_step",
                "source": "rag_proxy/memgraphrag/retrieval.py",
                "extracted_at": extracted_at,
            }
        )
        edges.append(
            {
                "from": comp_ids["MemGraphRetriever"],
                "to": sid,
                "type": "CONTAINS",
                "source": "rag_proxy/memgraphrag/retrieval.py",
                "extracted_at": extracted_at,
                "confidence": 1.0,
            }
        )
    for i in range(len(step_ids) - 1):
        edges.append(
            {
                "from": step_ids[i],
                "to": step_ids[i + 1],
                "type": "PRECEDES",
                "source": "rag_proxy/memgraphrag/retrieval.py",
                "extracted_at": extracted_at,
                "confidence": 1.0,
            }
        )

    return nodes, edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MemGraphRAG ontology graph")
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Optional memgraphrag.sqlite path for corpus ontology export",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Skip test-fixture sample instances",
    )
    args = parser.parse_args()

    extracted_at = now_iso()
    nodes, edges = build_meta_model(extracted_at)

    if not args.no_sample:
        add_sample_fixture(nodes, edges, extracted_at)

    db_path = args.db
    if db_path is None:
        default_db = Path("/var/lib/rag_proxy/memgraphrag.sqlite")
        if default_db.exists():
            db_path = default_db

    if db_path is not None:
        db_nodes, db_edges = export_sqlite(db_path, extracted_at)
        nodes.extend(db_nodes)
        edges.extend(db_edges)
        print(f"sqlite export from {db_path}: +{len(db_nodes)} nodes, +{len(db_edges)} edges")

    by_id: dict[str, dict] = {}
    for node in nodes:
        by_id[node["id"]] = node
    nodes = list(by_id.values())

    (KG / "nodes.jsonl").write_text(
        "\n".join(json.dumps(n, ensure_ascii=True) for n in sorted(nodes, key=lambda x: x["id"]))
        + "\n",
        encoding="utf-8",
    )
    (KG / "edges.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=True) for e in edges) + "\n",
        encoding="utf-8",
    )
    print(f"memgraphrag ontology: nodes={len(nodes)} edges={len(edges)}")


if __name__ == "__main__":
    main()
