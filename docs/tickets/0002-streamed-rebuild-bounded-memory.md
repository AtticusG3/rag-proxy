# 0002: Stream the rebuild: page-wise tokenize to temporary artifacts

**Parent:** docs/specs/0001-streamed-sparse-rebuild.md
**Blocked by:** 0001-injectable-qdrant-scroll (needs the fake-scroll seam)
**Status:** ready-for-agent

## What to build

The full rebuild stops holding the corpus in RAM: each scroll page is slimmed
and tokenized immediately, and tokenized batches are written to temporary
on-disk artifacts as they go; the CSC matrix and metadata are finalized from
those artifacts. End to end: an operator (or offline test) triggers a rebuild
of a large collection and the process memory stays bounded to roughly one
page batch, while the resulting snapshot returns the same top hits as the old
in-RAM rebuild. Small corpora keep using the existing fast path unchanged.

## Acceptance criteria

- [ ] A route-level test with a fake multi-page scroll shows only one page
      batch is retained at a time (observable via the injected provider or
      the temp-artifact growth pattern, not internal buffers).
- [ ] A rebuilt snapshot's search results (ids, order, scores) match the
      previous in-RAM rebuild for the same corpus, including the zero-score
      padding drop.
- [ ] Tokenization parity: vocabulary, token ids, and threading threshold
      behaviour are unchanged; the existing tokenizer parity test still
      passes.
- [ ] Interrupted rebuild's temporary artifacts are cleaned up, and a
      retry rebuilds from scratch safely.
- [ ] The live snapshot is untouched and keeps serving for the whole
      rebuild; the completed rebuild installs via the existing atomic swap.
