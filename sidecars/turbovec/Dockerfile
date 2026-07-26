# syntax=docker/dockerfile:1
# Build from repo root: docker compose build turbovec-index
FROM python:3.12-slim-bookworm

WORKDIR /app

RUN useradd --create-home --uid 10004 appuser

COPY sidecars/turbovec/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY sidecars/turbovec/core.py ./core.py
COPY sidecars/turbovec/app.py ./app.py

ENV TURBOVEC_HOST=0.0.0.0 \
    TURBOVEC_PORT=8097 \
    TURBOVEC_DIM=768 \
    TURBOVEC_BIT_WIDTH=4 \
    TURBOVEC_INDEX_PATH=/var/lib/rag_proxy/turbovec/index.tvim \
    QDRANT_URL=http://qdrant:6333 \
    QDRANT_COLLECTION=nomad_knowledge_base \
    PYTHONUNBUFFERED=1

RUN mkdir -p /var/lib/rag_proxy/turbovec && chown -R appuser:appuser /var/lib/rag_proxy

EXPOSE 8097

USER appuser

VOLUME ["/var/lib/rag_proxy/turbovec"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8097/health', timeout=5)"

CMD ["python", "app.py"]
