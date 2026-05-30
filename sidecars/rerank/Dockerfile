# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10002 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core.py app.py .

ENV RERANK_HOST=0.0.0.0 \
    RERANK_PORT=8095 \
    RERANK_MODEL=BAAI/bge-reranker-base \
    PYTHONUNBUFFERED=1

EXPOSE 8095

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8095/health', timeout=5)"

CMD ["python", "app.py"]
