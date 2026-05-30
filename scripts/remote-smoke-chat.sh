#!/usr/bin/env bash
# Smoke POST to temp proxy (needs LLAMA_SWAP API key in env if required).
set -euo pipefail
PORT="${1:-8087}"
AUTH="${OPENAI_API_KEY:-}"
HDR=(-H 'Content-Type: application/json')
if [ -n "$AUTH" ]; then
  HDR+=(-H "Authorization: Bearer $AUTH")
fi
curl -s -m 120 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
  "${HDR[@]}" \
  -d '{"model":"bonsai-8b","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":40,"stream":false}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('choices',[{}])[0].get('message',{}).get('content','ERR'))"
