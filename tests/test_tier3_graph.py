"""Tests for graph traversal depth behavior."""

from pathlib import Path

from rag_proxy.stages.tier3_graph import _ensure_schema, _query_graph


def test_query_graph_depth_traverses_multi_hop(tmp_path: Path):
    db_path = tmp_path / "graph.sqlite"
    _ensure_schema(db_path)

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO entities (id, kind, name) VALUES ('a', 'svc', 'nomad')")
        conn.execute("INSERT INTO entities (id, kind, name) VALUES ('b', 'svc', 'qdrant')")
        conn.execute("INSERT INTO entities (id, kind, name) VALUES ('c', 'svc', 'llama-swap')")
        conn.execute("INSERT INTO edges (src, dst, rel) VALUES ('a', 'b', 'depends_on')")
        conn.execute("INSERT INTO edges (src, dst, rel) VALUES ('b', 'c', 'depends_on')")

    depth1 = _query_graph(db_path, ["nomad"], max_depth=1)
    depth2 = _query_graph(db_path, ["nomad"], max_depth=2)

    assert any("nomad -[depends_on]-> svc:qdrant" in line for line in depth1)
    assert not any("qdrant -[depends_on]-> svc:llama-swap" in line for line in depth1)
    assert any("qdrant -[depends_on]-> svc:llama-swap" in line for line in depth2)
