# syntax=docker/dockerfile:1
# Build from repo root: docker compose build sparse-index
FROM python:3.12-slim-bookworm

WORKDIR /app

RUN useradd --create-home --uid 10003 appuser

COPY sidecars/sparse/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_proxy/chunk_text.py ./chunk_text.py
COPY sidecars/sparse/core.py ./core.py
COPY sidecars/sparse/app.py ./app.py

ENV SPARSE_HOST=0.0.0.0 \
    SPARSE_PORT=8096 \
    QDRANT_URL=http://qdrant:6333 \
    QDRANT_COLLECTION=nomad_knowledge_base \
    SPARSE_REFRESH_SEC=3600 \
    PYTHONUNBUFFERED=1

EXPOSE 8096

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8096/health', timeout=5)"

CMD ["python", "app.py"]
