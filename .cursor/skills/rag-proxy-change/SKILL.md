---
name: rag-proxy-change
description: >-
  Implements surgical changes to rag_proxy.py RAG injection, Qdrant search,
  embedding, or proxy passthrough. Use when modifying chat augmentation,
  env vars, CHAT_PATHS, payload field extraction, streaming relay, or fail-open
  error handling in this repo.
---

# rag_proxy — Code Change

## Before editing

1. Read `rag_proxy.py` sections: Configuration, RAG helpers, `proxy()` route.
2. Read `tests/test_rag_helpers.py` and `README.md` RAG behavior section.
3. State assumptions (e.g. which chat paths, payload schema, streaming vs buffered).

## Success criteria template

Copy and fill before coding:

```
- [ ] Behavior: <what users/clients see>
- [ ] Fail-open: RAG errors still forward original body
- [ ] pytest tests/ -q passes
- [ ] .env.example + module docstring updated if new env vars
```

## Change checklist

- [ ] Only touch lines required for the goal (Rule 3).
- [ ] No new abstractions unless used in 2+ places (Rule 2).
- [ ] Match existing: `log.warning` for recoverable failures, `log.info` for successful injection.
- [ ] RAG block stays inside `proxy()` try/except — never raise to client on RAG failure.
- [ ] Streaming: if touching upstream forward, preserve `relay_upstream` client lifetime.

## Common edit points

| Goal | Where |
|------|--------|
| New chat API path | `CHAT_PATHS` set |
| Chunk text field | `extract_chunk_text` key tuple |
| Query source | `extract_query_text` |
| Context format | `inject_context` |
| Retrieval params | `TOP_K`, `SIMILARITY_THRESHOLD`, `search_qdrant` |
| Embed robustness | `get_embedding`, `EMBED_*` env |

## After editing

```bash
cd <repo-root>
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux
pytest tests/ -q
```

If behavior is integration-level (live Qdrant/embed), document manual verification steps; do not add network calls to unit tests unless explicitly requested.
