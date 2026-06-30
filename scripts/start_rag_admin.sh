#!/usr/bin/env bash
# Start rag-admin with key=value env load (avoids executing bare commands in env files).
set -euo pipefail

REPO="${REPO_ROOT:-/opt/ai/repo/rag_proxy}"
ENV_FILE="${RAG_ADMIN_ENV_FILE:-/opt/ai/config/rag-admin.env}"
PYTHON=/opt/ai/venv/bin/python
LOG="${RAG_ADMIN_MANUAL_LOG:-/tmp/rag-admin-manual.log}"

export PYTHONPATH="$REPO"
cd "$REPO"

exec "$PYTHON" -c "
import os, subprocess, sys
from pathlib import Path

env_file = Path('$ENV_FILE')
if env_file.is_file():
    for line in env_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        os.environ.setdefault(key.strip(), value.strip())

pool = Path('/opt/ai/config/nomic-embed-pool.env')
if pool.is_file():
    for line in pool.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        os.environ.setdefault(key.strip(), value.strip())

os.chdir('$REPO')
sys.exit(subprocess.call([sys.executable, '-m', 'rag_admin']))
" >> "$LOG" 2>&1 &

sleep 2
pgrep -af "python -m rag_admin" || { echo "[X] rag-admin failed to start; tail $LOG"; tail -20 "$LOG"; exit 1; }
echo "[ok] rag-admin started"
