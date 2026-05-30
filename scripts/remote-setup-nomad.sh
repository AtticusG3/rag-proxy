#!/usr/bin/env bash
# Run on nomad after tarball extract (localhost stack).
# Dev instance: PROXY_PORT=8087 (production rag-proxy.service keeps 8088).
set -euo pipefail
cd "$(dirname "$0")/.."

cat > .env <<'EOF'
LLAMA_SWAP_URL=http://127.0.0.1:8080
EMBED_URL=http://127.0.0.1:8089
QDRANT_URL=http://192.168.1.36:6333
QDRANT_COLLECTION=nomad_knowledge_base
TOP_K=5
SIMILARITY_THRESHOLD=0.65
PROXY_HOST=0.0.0.0
PROXY_PORT=8087
LOG_LEVEL=INFO
ENABLE_COGNITIVE_PIPELINE=false
ENABLE_METRICS=true
EOF

python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q -r requirements.txt pytest
set -a
# shellcheck disable=SC1091
. ./.env
set +a
python -m pytest tests/ -q

# shellcheck source=dev-log-cap.sh
source "$(dirname "$0")/dev-log-cap.sh"
rotate_dev_log_for_restart

fuser -k "${PROXY_PORT}/tcp" 2>/dev/null || true
sleep 2
nohup python rag_proxy.py > /tmp/rag_proxy_test.log 2>&1 &
sleep 3

echo "[metrics]"
curl -s "http://127.0.0.1:${PROXY_PORT}/metrics" | head -5

echo "[embed]"
curl -s -m 10 -X POST http://127.0.0.1:8089/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-text-v1.5","input":"hello"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('embed_dims', len(d['data'][0]['embedding']))"

echo "[log]"
tail -20 /tmp/rag_proxy_test.log
