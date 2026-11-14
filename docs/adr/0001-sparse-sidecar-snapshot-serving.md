# Sparse sidecar serves from a snapshot; rebuild is a streamed exception

The sparse sidecar must answer search on a multi-million-chunk collection on a
small homelab host. We decided that search is always served from the persisted
snapshot (memory-mapped CSC, ~0.75G RSS), never from a live in-memory index,
and that a full rebuild is the exceptional path: it streams the Qdrant scroll
page by page into temporary artifacts and only atomically swaps the snapshot
directory once the replacement is complete. While a rebuild runs, the old
snapshot keeps serving; if no snapshot exists, `/search` reports a starting
state rather than blocking.

Why: an earlier design rebuilt the full index in RAM before installing it,
peaking near the old heap size (~24.9G for the 2.11M-doc collection) and
risking OOM on first boot or `POST /reindex`. Snapshot serving made steady
state cheap; keeping the rebuild peak bounded closes the remaining exposure.
The trade-off: streaming rebuild is more code than building in place, and a
partially-written rebuild is wasted work if the process dies — acceptable,
because the old snapshot remains valid and a retry is idempotent.

## Considered Options

- In-RAM rebuild then install (previous behaviour): simplest code, unbounded
  peak memory — rejected as the OOM risk is real on the production host.
- Incremental (delta) updates from ingest: cheaper steady state, but needs an
  ingest→sidecar notification contract across processes; deferred.

## Consequences

- Rebuild code must never hold the whole corpus (texts or token lists) in
  memory at once; per-page batches only.
- `/health` must distinguish `starting` (no snapshot yet) from `ready` from
  `rebuilding`, and failed rebuilds must leave the previous snapshot intact.
