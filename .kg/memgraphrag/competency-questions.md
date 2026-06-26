# Competency Questions — MemGraphRAG Ontology

Graph: `memgraphrag-ontology`
Scope: Three-layer memory (schema / fact / passage), extraction entity classes, and corpus ontology instances.

## Must answer (P0)

| ID | Type | Question | Query pattern |
|----|------|----------|---------------|
| CQ-M01 | SCQ | What memory layers exist? | Nodes where `type=MgrLayer` |
| CQ-M02 | SCQ | What entity classes can LLM extraction assign? | Nodes where `type=MgrEntityClass` |
| CQ-M03 | FCQ | What is the root ontology node? | Lookup `mgr-root:memgraphrag` |
| CQ-M04 | VCQ | Are all edge endpoints valid? | `validate.py` pass |

## Should answer (P1)

| ID | Type | Question | Query pattern |
|----|------|----------|---------------|
| CQ-M05 | RCQ | What ontology patterns exist in the index? | Nodes where `type=MgrOntologyPattern` |
| CQ-M06 | RCQ | Which facts conform to each pattern? | `CONFORMS_TO` inbound to pattern |
| CQ-M07 | RCQ | Can every fact be traced to a passage? | `EXTRACTED_IN` from fact |
| CQ-M08 | RCQ | What code components implement MemGraphRAG? | Nodes where `type=MgrComponent` |
| CQ-M09 | RCQ | What is the online retrieval step order? | Traverse `PRECEDES` from first step |

## Verification log

| ID | Status | Notes |
|----|--------|-------|
| CQ-M01 | pass | See `queries/cq-results.md` |
| CQ-M02 | pass | |
| CQ-M03 | pass | |
| CQ-M04 | pass | `validate.py` |
| CQ-M05 | pass | |
| CQ-M06 | pass | |
| CQ-M07 | pass | |
| CQ-M08 | pass | |
| CQ-M09 | pass | |
