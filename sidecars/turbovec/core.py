"""TurboQuant dense index helpers (no FastAPI dependency).

Qdrant point ids are 32-character lowercase MD5 hex strings. IdMapIndex keys are
uint64; ``hex_id_to_u64`` maps the first 16 hex digits (64 bits) into that key.
The companion SQLite ``id_map`` table stores the full 32-char hex for each u64 so
search can return the exact Qdrant id. If ``id_map`` is missing a row (stale index,
partial wipe), hits for that u64 are skipped rather than guessing from the prefix.

After Qdrant ``clear_collection`` or any full collection drop, TurboVec on disk is
out of sync until the operator ``POST /reindex`` on the sidecar (scroll Qdrant).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("turbovec-sidecar")

try:
    from turbovec import IdMapIndex

    HAS_TURBOVEC = True
except ImportError:  # pragma: no cover - optional at import for offline unit tests
    IdMapIndex = None  # type: ignore[misc, assignment]
    HAS_TURBOVEC = False

DEFAULT_DIM = 768
DEFAULT_BIT_WIDTH = 4


def hex_id_to_u64(hex_id: str) -> int:
    """Map Qdrant MD5 hex point id to uint64 for IdMapIndex (first 16 hex digits)."""
    cleaned = str(hex_id).strip().lower()
    if len(cleaned) < 16:
        raise ValueError(f"hex id too short for u64 map: {hex_id!r}")
    return int(cleaned[:16], 16)


def normalize_qdrant_hex_id(hex_id: str) -> str:
    """Require canonical 32-char hex as produced by ingest (MD5 point ids)."""
    cleaned = str(hex_id).strip().lower()
    if len(cleaned) != 32:
        raise ValueError(f"hex id must be 32 chars (Qdrant MD5): {hex_id!r}")
    int(cleaned, 16)
    return cleaned


class IdStore:
    """Persist u64 <-> hex id mapping beside the .tvim index."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS id_map ("
            "u64 TEXT PRIMARY KEY NOT NULL, "
            "hex_id TEXT NOT NULL UNIQUE)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS id_map_staging ("
            "u64 TEXT PRIMARY KEY NOT NULL, "
            "hex_id TEXT NOT NULL UNIQUE)"
        )
        self._conn.execute("DELETE FROM id_map_staging")
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def put_many(self, pairs: list[tuple[int, str]]) -> None:
        if not pairs:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO id_map (u64, hex_id) VALUES (?, ?)",
                [(str(uid), hid) for uid, hid in pairs],
            )
            self._conn.commit()

    def delete_many(self, u64_ids: list[int]) -> None:
        if not u64_ids:
            return
        with self._lock:
            self._conn.executemany(
                "DELETE FROM id_map WHERE u64 = ?",
                [(str(uid),) for uid in u64_ids],
            )
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM id_map")
            self._conn.commit()

    def stage_many(self, pairs: list[tuple[int, str]]) -> None:
        """Buffer rebuild rows; they replace id_map only on commit_staged()."""
        if not pairs:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO id_map_staging (u64, hex_id) VALUES (?, ?)",
                [(str(uid), hid) for uid, hid in pairs],
            )
            self._conn.commit()

    def clear_staged(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM id_map_staging")
            self._conn.commit()

    def commit_staged(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM id_map")
            self._conn.execute(
                "INSERT INTO id_map (u64, hex_id) SELECT u64, hex_id FROM id_map_staging"
            )
            self._conn.execute("DELETE FROM id_map_staging")

    def hex_for_u64(self, u64_id: int) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT hex_id FROM id_map WHERE u64 = ?",
                (str(int(u64_id)),),
            ).fetchone()
        return str(row[0]) if row else None

    def hex_for_u64_many(self, u64_ids: list[int]) -> dict[int, str]:
        if not u64_ids:
            return {}
        keys = list(dict.fromkeys(int(uid) for uid in u64_ids))
        placeholders = ",".join("?" * len(keys))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT u64, hex_id FROM id_map WHERE u64 IN ({placeholders})",
                [str(k) for k in keys],
            ).fetchall()
        return {int(row[0]): str(row[1]) for row in rows}

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM id_map").fetchone()
        return int(row[0]) if row else 0


class TurboIndex:
    """Thread-safe IdMapIndex + hex id store."""

    def __init__(
        self,
        *,
        dim: int = DEFAULT_DIM,
        bit_width: int = DEFAULT_BIT_WIDTH,
        index_path: Path,
        id_db_path: Path | None = None,
    ) -> None:
        if not HAS_TURBOVEC:
            raise RuntimeError("turbovec package is not installed")
        if bit_width not in (2, 3, 4):
            raise ValueError("bit_width must be 2, 3, or 4")
        if dim % 8 != 0 or dim <= 0:
            raise ValueError("dim must be a positive multiple of 8")
        self.dim = dim
        self.bit_width = bit_width
        self.index_path = Path(index_path)
        self.id_db_path = Path(id_db_path) if id_db_path else self.index_path.with_suffix(
            ".ids.sqlite"
        )
        self._lock = threading.Lock()
        self._ids = IdStore(self.id_db_path)
        self._index = IdMapIndex(dim=dim, bit_width=bit_width)

    def close(self) -> None:
        self._ids.close()

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    def load(self) -> bool:
        """Load .tvim from disk if present. Returns True when loaded."""
        if not self.index_path.is_file():
            return False
        loaded = IdMapIndex.load(str(self.index_path))
        with self._lock:
            self._index = loaded
            if loaded.dim is not None:
                self.dim = int(loaded.dim)
            self.bit_width = int(loaded.bit_width)
        log.info(
            "loaded turbovec index path=%s vectors=%d dim=%s bit_width=%s",
            self.index_path,
            len(self),
            self.dim,
            self.bit_width,
        )
        return True

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._index.write(str(self.index_path))
        log.info("saved turbovec index path=%s vectors=%d", self.index_path, len(self))

    def reset(self) -> None:
        with self._lock:
            self._index = IdMapIndex(dim=self.dim, bit_width=self.bit_width)
            self._ids.clear()

    def _prepare(
        self, hex_ids: list[str], vectors: list[list[float]]
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
        if len(hex_ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        canonical = [normalize_qdrant_hex_id(h) for h in hex_ids]
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"vectors must be shape (n, {self.dim}), got {arr.shape}")
        u64_ids = np.asarray([hex_id_to_u64(h) for h in canonical], dtype=np.uint64)
        pairs = [(int(u64_ids[i]), canonical[i]) for i in range(len(canonical))]
        return arr, u64_ids, pairs

    @staticmethod
    def _add_deduped(index: Any, arr: np.ndarray, u64_ids: np.ndarray) -> None:
        for uid in u64_ids.tolist():
            if int(uid) in index:
                index.remove(int(uid))
        index.add_with_ids(arr, u64_ids)

    def add(self, hex_ids: list[str], vectors: list[list[float]]) -> int:
        if not hex_ids:
            return 0
        arr, u64_ids, pairs = self._prepare(hex_ids, vectors)
        with self._lock:
            self._add_deduped(self._index, arr, u64_ids)
            self._ids.put_many(pairs)
        return len(hex_ids)

    def rebuild(self) -> "TurboRebuild":
        """Start a full rebuild; the live index keeps serving until commit()."""
        return TurboRebuild(self)

    def remove(self, hex_ids: list[str]) -> int:
        if not hex_ids:
            return 0
        removed = 0
        u64_list: list[int] = []
        with self._lock:
            for hid in hex_ids:
                uid = hex_id_to_u64(hid)
                if self._index.remove(uid):
                    removed += 1
                u64_list.append(uid)
            self._ids.delete_many(u64_list)
        return removed

    def search(
        self,
        vector: list[float],
        *,
        limit: int,
        score_threshold: float | None = None,
        allowlist: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """ANN top-k first; ``score_threshold`` filters returned hits (not index-side)."""
        limit = max(1, limit)
        query = np.asarray([vector], dtype=np.float32)
        if query.shape[1] != self.dim:
            raise ValueError(f"query dim {query.shape[1]} != index dim {self.dim}")
        allow_u64: np.ndarray | None = None
        if allowlist is not None:
            if not allowlist:
                return []
            allow_u64 = np.asarray(
                [hex_id_to_u64(h) for h in allowlist],
                dtype=np.uint64,
            )
        with self._lock:
            if len(self._index) == 0:
                return []
            if allow_u64 is not None:
                scores, ids = self._index.search(query, limit, allowlist=allow_u64)
            else:
                scores, ids = self._index.search(query, limit)
            flat_scores = [float(s) for s in scores[0].tolist()]
            flat_ids = [int(i) for i in ids[0].tolist()]
            hex_map = self._ids.hex_for_u64_many(flat_ids)

        results: list[dict[str, Any]] = []
        for score, uid in zip(flat_scores, flat_ids):
            if score_threshold is not None and score < score_threshold:
                continue
            hex_id = hex_map.get(uid)
            if hex_id is None:
                log.warning("missing hex id for u64=%s; skipping hit", uid)
                continue
            results.append({"id": hex_id, "score": score})
        return results

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n = len(self._index)
            dim = self._index.dim
        return {
            "vectors": n,
            "dim": dim if dim is not None else self.dim,
            "bit_width": self.bit_width,
            "id_map_rows": self._ids.count(),
            "index_path": str(self.index_path),
        }


class TurboRebuild:
    """Off-to-the-side full rebuild.

    Resetting in place leaves the sidecar answering every search with zero hits
    for the whole reindex, so vectors accumulate here and swap in atomically.
    """

    def __init__(self, owner: TurboIndex) -> None:
        self._owner = owner
        self._index = IdMapIndex(dim=owner.dim, bit_width=owner.bit_width)
        self._count = 0
        owner._ids.clear_staged()

    def add(self, hex_ids: list[str], vectors: list[list[float]]) -> int:
        if not hex_ids:
            return 0
        arr, u64_ids, pairs = self._owner._prepare(hex_ids, vectors)
        TurboIndex._add_deduped(self._index, arr, u64_ids)
        self._owner._ids.stage_many(pairs)
        self._count = len(self._index)
        return len(hex_ids)

    def commit(self) -> int:
        with self._owner._lock:
            self._owner._index = self._index
            self._owner._ids.commit_staged()
        return self._count

    def abort(self) -> None:
        self._owner._ids.clear_staged()
