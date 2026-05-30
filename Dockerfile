# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /var/lib/rag_proxy \
    && chown appuser:appuser /var/lib/rag_proxy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_proxy.py rag_proxy/ VERSION ./

ENV PROXY_HOST=0.0.0.0 \
    PROXY_PORT=8088 \
    PYTHONUNBUFFERED=1

EXPOSE 8088

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8088/v1/models', timeout=4)"

CMD ["python", "rag_proxy.py"]
