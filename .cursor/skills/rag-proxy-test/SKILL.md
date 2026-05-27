---
name: rag-proxy-test
description: >-
  Adds or fixes pytest tests for rag_proxy helpers using offline import pattern.
  Use when writing tests, improving coverage, or when the user mentions test
  intent, business rules, extract_query_text, or inject_context.
---

# rag_proxy — Testing

## Principles (Rule 7)

Tests must encode **why** behavior matters:

- **Good**: Assert last user message wins because RAG should answer the latest turn, not an earlier question.
- **Bad**: Assert `len(out) == 2` with no link to product rule.

A test should fail if someone changes business logic (e.g. uses first user message instead of last) even if the function still "works."

## Test layout

- File: `tests/test_rag_helpers.py`
- Import `rag_proxy.py` via `importlib` (existing pattern) — keeps tests offline.
- Test only pure helpers unless explicitly asked for HTTP integration tests.

## Workflow

1. Identify the business rule (one sentence).
2. Write test name as the rule: `test_extract_query_text_uses_last_user_message`.
3. Implement minimal assertion that breaks if the rule is violated.
4. Run: `pytest tests/ -q`

## Adding a test for new helper

```python
def test_<behavior>_<outcome>():
    # Arrange: minimal messages/chunks
    # Act
    # Assert: outcome tied to WHY (role, content substring, order)
```

Do not mock httpx/FastAPI for helper tests. For new proxy-route behavior, prefer extracting a pure function and testing that (Rule 2).

## Anti-patterns

- Snapshot-testing entire injected system prompts without asserting invariant phrases.
- Tests that pass if return type is `list` but content is wrong.
- Live calls to `EMBED_URL` or `QDRANT_URL` in default unit tests.
