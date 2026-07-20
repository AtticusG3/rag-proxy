"""remove_file_from_index must scrub dense, disk, BM25, and MemGraphRAG."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

from ingest.db import IngestDatabase
from ingest.worker import IngestConfig, IngestWorker


def test_remove_file_from_index_deletes_disk_qdrant_state_and_memgraph() -> None:
    """Document removal is complete: file gone, dense deleted, BM25 reindexed, MemGraph scrubbed."""
    with tempfile.TemporaryDirectory() as zim_dir:
        upload_dir = tempfile.mkdtemp()
        db_path = os.path.join(zim_dir, "admin.sqlite")
        db = IngestDatabase(db_path)
        file_path = os.path.join(zim_dir, "doc.zim")
        part_path = f"{file_path}.part"
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("zim")
        with open(part_path, "w", encoding="utf-8") as handle:
            handle.write("partial")
        db.upsert_file_state(file_path, status="indexed")

        mem_db = os.path.join(zim_dir, "memgraphrag.sqlite")
        with open(mem_db, "w", encoding="utf-8") as handle:
            handle.write("")  # existence gate for scrub; load_memory is mocked

        config = IngestConfig(
            zim_dir=zim_dir,
            upload_dir=upload_dir,
            embed_url="http://127.0.0.1:1",
            qdrant_url="http://127.0.0.1:1",
            qdrant_collection="test",
            sparse_index_url="http://127.0.0.1:8096",
            memgraphrag_db_path=mem_db,
        )
        worker = IngestWorker(config, db)
        memory = MagicMock()
        memory.remove_passages_by_chunk_ids.return_value = 2

        with (
            patch("ingest.worker.list_point_ids_by_source", return_value=["p1", "p2"]),
            patch("ingest.worker.delete_by_source") as delete_dense,
            patch("ingest.worker.trigger_sparse_reindex") as sparse,
            patch(
                "rag_proxy.memgraphrag.memory.load_memory",
                return_value=memory,
            ) as load_mem,
        ):
            worker.remove_file_from_index(file_path)

        load_mem.assert_called_once_with(mem_db)
        memory.remove_passages_by_chunk_ids.assert_called_once_with({"p1", "p2"})
        memory.save.assert_called_once_with(mem_db)
        delete_dense.assert_called_once_with(
            config.qdrant_url, config.qdrant_collection, file_path
        )
        sparse.assert_called_once_with(config)
        assert not os.path.isfile(file_path)
        assert not os.path.isfile(part_path)
        assert db.get_file_state(file_path) is None
