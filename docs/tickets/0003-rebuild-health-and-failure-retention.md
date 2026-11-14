# 0003: Rebuild health states, progress, and failure retention

**Parent:** docs/specs/0001-streamed-sparse-rebuild.md
**Blocked by:** 0002-streamed-rebuild-bounded-memory (streams must exist to report on)
**Status:** ready-for-agent

## What to build

Operators can see what a rebuild is doing and trust it not to break serving.
`/health` gains a `rebuilding` state with page/document progress counters
alongside the existing `starting` and ready states. A rebuild that fails at
any point (scroll error, tokenize error, finalize error) deletes its
temporary artifacts, leaves the previous snapshot serving, and the reindex
endpoint returns an error result instead of hanging or half-installing. End
to end: an offline test drives a failing fake scroll and observes the error
result plus a still-serving old snapshot; a slow fake scroll shows progress
counters in `/health`.

## Acceptance criteria

- [ ] `/health` reports `starting`, `ready`, and `rebuilding` (with page and
      document progress) at the right times during a fake-scroll rebuild.
- [ ] A scroll/tokenize/finalize failure returns an error result from the
      reindex endpoint, leaves no temporary artifacts, and leaves the
      previous snapshot installed and serving.
- [ ] Concurrent same-collection rebuild attempts are rejected by the
      existing per-collection lock (no regression).
- [ ] Search continues to answer from the old snapshot during both progress
      and failure scenarios in the tests.
