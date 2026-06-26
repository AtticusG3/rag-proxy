# CQ Results — MemGraphRAG Ontology

Generated: 2026-06-26T10:58:06Z

| ID | Status | Result |
|----|--------|--------|
| CQ-M01 | pass | 3 layers: schema, fact, passage |
| CQ-M02 | pass | 8 closed NER classes from extraction prompt |
| CQ-M03 | pass | `mgr-root:memgraphrag` |
| CQ-M04 | pass | 0 violations (25 nodes, 36 edges) |
| CQ-M05 | pass | 1 sample pattern (Person-knows-Person) |
| CQ-M06 | pass | 1 fact conforms to pattern |
| CQ-M07 | pass | fact traced to passage chunk-1 |
| CQ-M08 | pass | 4 components (memory, retriever, indexer, stage) |
| CQ-M09 | pass | 4-step retrieval: score -> rerank -> PPR -> aggregate |

Refresh: `.venv/Scripts/python.exe .kg/memgraphrag/extract.py`

Corpus export (when index exists on nomad):

`.venv/Scripts/python.exe .kg/memgraphrag/extract.py --db /var/lib/rag_proxy/memgraphrag.sqlite`
