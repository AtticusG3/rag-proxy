#!/usr/bin/env bash
# Run on clanker after tarball extract. Uses nomad stack via tailscale/LAN.
set -euo pipefail
cd "$(dirname "$0")/.."

cat > .env <<'EOF'
# Clanker temp deploy - endpoints from nomad production rag-proxy
LLAMA_SWAP_URL=http://nomad:8080
EMBED_URL=http://nomad:8089
QDRANT_URL=http://192.168.1.36:6333
QDRANT_COLLECTION=nomad_knowledge_base
TOP_K=5
SIMILARITY_THRESHOLD=0.65
PROXY_HOST=0.0.0.0
PROXY_PORT=8087
LOG_LEVEL=INFO
EMBED_MAX_CHARS=2000
EMBED_RETRIES=2
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

echo "[reachability]"
curl -s -m 5 -o /dev/null -w 'llama-swap:%{http_code}\n' http://nomad:8080/v1/models || true
curl -s -m 5 -o /dev/null -w 'qdrant:%{http_code}\n' http://192.168.1.36:6333/collections/nomad_knowledge_base || true
