# Spec: Streamed sparse-sidecar rebuild with bounded memory

Parent: grill output for the "next slice" question (issue #43). ADR:
[0001-sparse-sidecar-snapshot-serving](../adr/0001-sparse-sidecar-snapshot-serving.md).

## Problem Statement

An operator running the sparse sidecar against a large collection (2.11M
chunks, ~5.1G snapshot) has cheap steady-state search (~0.75G RSS) because
search serves from a memory-mapped snapshot — but the moment the sidecar first
boots, or the operator calls the reindex endpoint, the sidecar rebuilds the
whole BM25 index in RAM and peaks near ~25G before installing the snapshot.
On the production host this is an OOM risk: first boot can kill the sidecar
(and anything else on the box) precisely when the host is least able to spare
the memory.

## Solution

The rebuild path streams the Qdrant scroll page by page and never holds the
whole corpus — texts, payloads, or token lists — in memory. Each page is
tokenized immediately, tokenized batches are written to temporary on-disk
artifacts, and only at finalize time are the snapshot artifacts assembled and
atomically swapped into place. The old snapshot keeps serving search for the
entire rebuild; a failed or interrupted rebuild leaves the previous snapshot
intact and reports failure; `/health` shows `rebuilding` with progress while
it runs.

## User Stories

1. As an operator, I want a rebuild to run in bounded memory, so that first
   boot on a large collection cannot OOM the host.
2. As an operator, I want the previous snapshot to keep serving search during
   a rebuild, so that clients see no gap.
3. As an operator, I want a failed rebuild to leave the existing snapshot
   untouched, so that a crash never loses serving capability.
4. As an operator, I want `/health` to report `rebuilding` and progress
   (pages/documents seen), so that I can distinguish a stuck rebuild from a
   working one.
5. As an operator, I want the swap of the new snapshot to be atomic, so that
   search never observes a half-written index.
6. As an operator, I want a rebuild retry to be safe after an interrupted one,
   so that recovery needs no manual cleanup of temp artifacts.
7. As a developer, I want the Qdrant scroll to be an injected dependency at
   the sidecar route seam, so that rebuild behaviour is testable offline
   against a fake scroll.
8. As a developer, I want the existing tokenizer and score behaviour preserved
   byte-for-byte, so that a rebuilt snapshot returns the same top hits as the
   old in-RAM path.
9. As an operator, I want the reindex endpoint to return a failure result (not
   hang) when Qdrant scrolls fail, so that automation can retry.
10. As an operator on a small test corpus, I want small rebuilds to stay
    simple and fast, so that dev usage does not pay the streaming machinery
    beyond what it needs.

## Implementation Decisions

- Rebuild runs as a page loop over the Qdrant scroll: fetch page → slim
  payloads → tokenize batch → append tokenized output to temporary artifacts
  → drop the batch. No stage holds the full corpus.
- Tokenized batches are persisted to temporary files during the loop; the
  CSC matrix and metadata are finalized from those artifacts, not from an
  in-memory corpus.
- The existing vocabulary/tokenization path (whole-word pattern, English
  stopwords dropped, thread-pooled split above the parallel threshold) is
  reused unchanged; streaming must not change which tokens exist or their ids.
- Installation of a completed rebuild stays as it is today: build under a
  temporary directory next to the live snapshot, then atomic directory
  replace. A concurrent same-collection rebuild is rejected by the existing
  per-collection lock.
- A rebuild that fails at any point deletes its temporary artifacts and
  leaves the live snapshot directory and the registry state unchanged; the
  endpoint returns an error result.
- `/health` gains a `rebuilding` state with page/document progress counters
  alongside the existing `starting` and ready states.
- The Qdrant scroll becomes an injectable async page provider used by the
  route handlers; production wires the current Qdrant client, tests wire a
  fake. No new retrieval logic.
- Scope is the sparse sidecar only. TurboVec's RAM-copying index load is a
  third-party library limitation and is explicitly out of scope.

## Testing Decisions

- Good tests exercise the sidecar's external behaviour through its public
  surface: the FastAPI routes (`/reindex`, `/health`, `/search`) with a fake
  Qdrant scroll, plus the core index objects for tokenization parity — never
  internal buffers or file layouts.
- Module under test: the sparse sidecar (app routes + core index/registry).
- Prior art: `tests/test_sparse_index.py` (core index, mmap round-trip,
  registry install) and `tests/test_sidecar_contract.py` (route-level sidecar
  tests). New tests follow these patterns; no network — the fake scroll is an
  in-process async provider.

## Test Seam

**Chosen seam: the existing sidecar HTTP route seam** (`/reindex` + `/health`
+ `/search`) with an injected fake Qdrant scroll provider, supplemented by the
existing `SparseIndex`/`IndexRegistry` core seam for tokenization parity.

Why: both seams already exist — routes are real FastAPI endpoints and the
core objects are already unit-tested — so no new seam is introduced. The route
seam is the highest seam available: it verifies end-to-end behaviour
(rebuild → swap → serve, failure → old snapshot intact) exactly as an operator
experiences it, while the injected scroll keeps the suite offline per the
repo's no-network rule. Building a lower seam (e.g. a page-iterator class)
would test machinery rather than behaviour.

## Out of Scope

- TurboVec sidecar memory behaviour (third-party library limitation).
- Incremental/delta updates from ingest (needs a cross-process notification
  contract; see ADR consequences).
- Changes to the proxy, ingest worker, admin UI, or any other sidecar.
- Filing issues: this run writes files only; the repo manager publishes them.
- Any code, config, or CI change in this branch (docs-only per the ticket).

## Further Notes

### Assumptions (grill answered without a human; each decision self-made)

1. **Subject**: the review-tracking issue body was unreachable (no API token
   in this environment), so the grill used the repo's current state —
   specifically the documented production figures in the agent guide (rebuild
   peak ~24.9G vs ~0.75G steady state, TurboVec 8G RSS) — as the subject.
2. **Slice choice**: three candidates were weighed — streamed sparse rebuild;
   TurboVec mmap (rejected: third-party library limitation, not repo code);
   deflaking the two known-red tests (rejected: small, and #21/#22 already
   have issues). The sparse rebuild peak is the largest owner-owned risk.
3. **Grill answers**: with no human available, every question was answered
   with the recommended option; no recommendation was overridden. The design
   tree is flat: subject → approach → seam → scope.
4. **Glossary**: this repo had no `GLOSSARY.md`; it was created with the
   language this spec uses plus the core pipeline vocabulary.
5. **ADR**: exactly one decision met the bar (hard to reverse, surprising,
   real trade-off) and is recorded as ADR-0001; the tokenization-parity
   decision did not (it is "keep existing behaviour", not a choice).
6. **Ticket numbering/parents**: spec and ticket files are numbered from
   0001; ticket `Parent:` lines reference this spec file, not a Gitea number,
   because this run cannot file issues.
7. **Small-corpus behaviour**: the existing below-threshold single-thread
   path is preserved as-is; streaming must not slow small rebuilds measurably
   (the threshold heuristic stays).

### Verification sketch

A rebuild of the current production-sized collection is not reproducible in
unit tests; verification uses a fake scroll with thousands of pages and
asserts (a) peak retained batch count stays at one page, (b) top hits equal
the old path's for the same corpus, (c) `/health` transitions and failure
retention. Production-sized validation is an operator follow-up, not a unit
test.
