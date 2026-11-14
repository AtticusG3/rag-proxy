# 0001: Make the Qdrant scroll an injectable page provider (prefactor)

**Parent:** docs/specs/0001-streamed-sparse-rebuild.md
**Blocked by:** None (can start immediately)
**Status:** ready-for-agent

## What to build

The sparse sidecar's rebuild path currently reaches out to Qdrant itself, so
rebuild behaviour can only be tested against a live server. This ticket makes
the scroll an injectable async page provider at the sidecar route seam:
production wires the existing Qdrant client exactly as today, and tests can
supply a fake scroll. End to end: an offline test drives the sidecar's
reindex route against a fake multi-page scroll and observes the same
result as today — no behaviour change, no memory change yet.

## Acceptance criteria

- [ ] The sidecar routes (or the collection-sync path) receive their Qdrant
      page source as an injected async provider; the production default is
      behaviour-identical to today.
- [ ] An offline test drives a rebuild via the route with a fake scroll
      serving multiple pages and passes.
- [ ] No change to tokenization, scoring, snapshot layout, or memory profile.
- [ ] No network in tests; follows the patterns in the existing sidecar
      contract and sparse index tests.
