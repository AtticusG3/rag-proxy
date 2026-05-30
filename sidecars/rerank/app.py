#!/usr/bin/env python3
"""CPU cross-encoder rerank sidecar for rag_proxy (POST /rerank)."""

from __future__ import annotations

import logging
import os
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from core import rank_indices

log = logging.getLogger("rerank-sidecar")

MODEL_NAME = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
HOST = os.getenv("RERANK_HOST", "0.0.0.0")
PORT = int(os.getenv("RERANK_PORT", "8095"))

_encoder: Any = None


def get_encoder() -> Any:
    global _encoder
    if _encoder is None:
        from sentence_transformers import CrossEncoder

        log.info("Loading rerank model %s", MODEL_NAME)
        _encoder = CrossEncoder(MODEL_NAME)
        log.info("Rerank model ready")
    return _encoder


class Pair(BaseModel):
    query: str
    document: str


class RerankRequest(BaseModel):
    pairs: list[Pair] = Field(default_factory=list)
    top_k: int = 5


app = FastAPI(title="RAG Rerank Sidecar", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_NAME, "port": PORT}


@app.post("/rerank")
def rerank(body: RerankRequest) -> dict[str, list[int]]:
    if not body.pairs:
        return {"indices": []}

    encoder = get_encoder()
    sentences = [[pair.query, pair.document] for pair in body.pairs]
    raw_scores = encoder.predict(sentences)
    scores = [float(s) for s in raw_scores]
    return {"indices": rank_indices(scores, body.top_k)}


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("Rerank sidecar listening on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
