#!/usr/bin/env bash
set -uo pipefail
UNIT=/etc/systemd/system/sparse-sidecar.service

echo "=== backup + current cap ==="
cp "$UNIT" /tmp/sparse-sidecar.service.bak
grep -n 'SPARSE_MAX_POINTS' "$UNIT"

sed 's/^Environment=SPARSE_MAX_POINTS=.*/Environment=SPARSE_MAX_POINTS=0/' "$UNIT" > /tmp/sparse-sidecar.service.new
echo "=== new value ==="
grep -n 'SPARSE_MAX_POINTS' /tmp/sparse-sidecar.service.new

sudo -n /usr/bin/tee "$UNIT" < /tmp/sparse-sidecar.service.new >/dev/null && echo "[ok] unit written" || { echo "[FAIL] write"; exit 1; }
sudo -n /opt/ai/bin/nomic-pool-systemctl daemon-reload && echo "[ok] daemon-reload"

echo "=== verify systemd sees it ==="
systemctl show sparse-sidecar.service -p Environment --no-pager | tr ' ' '\n' | grep SPARSE_MAX_POINTS

echo "=== restart -> full corpus rebuild ==="
sudo -n /opt/ai/bin/nomic-pool-systemctl restart sparse-sidecar.service && echo "[ok] restarted"

echo "=== monitoring (projected peak ~20 GB, abort if available < 6 GB) ==="
for i in $(seq 1 90); do
  sleep 20
  MEM=$(systemctl show sparse-sidecar.service -p MemoryCurrent --value 2>/dev/null)
  RSS=$((MEM/1024/1024))
  AVAIL=$(free -m | awk '/^Mem:/{print $7}')
  H=$(curl -sf -m 5 http://127.0.0.1:18096/health 2>/dev/null)
  DOCS=$(echo "$H" | grep -o '"docs":[0-9]*' | cut -d: -f2)
  echo "t=$((i*20))s rss=${RSS}MB avail=${AVAIL}MB docs=${DOCS:-building}"
  if [[ "$AVAIL" -lt 6000 ]]; then
    echo "!! LOW MEMORY - aborting rebuild, reverting cap"
    sudo -n /usr/bin/tee "$UNIT" < /tmp/sparse-sidecar.service.bak >/dev/null
    sudo -n /opt/ai/bin/nomic-pool-systemctl daemon-reload
    sudo -n /opt/ai/bin/nomic-pool-systemctl restart sparse-sidecar.service
    echo "REVERTED"
    exit 1
  fi
  if [[ -n "$DOCS" && "$DOCS" -gt 200000 ]]; then
    echo "[ok] full index installed: $DOCS docs"
    break
  fi
done

echo "=== final ==="
curl -sf -m 10 http://127.0.0.1:18096/health; echo
MEM=$(systemctl show sparse-sidecar.service -p MemoryCurrent --value)
echo "sparse RSS: $((MEM/1024/1024)) MB"
free -m | head -2
echo "=== search smoke ==="
curl -sf -m 20 -X POST http://127.0.0.1:18096/search -H 'Content-Type: application/json' \
  -d '{"query":"nomic embed vector quantisation","limit":2}' | head -c 300; echo
echo "=== errors ==="
journalctl -u sparse-sidecar --no-pager --since '25 minutes ago' 2>/dev/null | grep -iE 'error|traceback|memoryerror' | tail -5 || echo none
