#!/usr/bin/env bash
# Run on nomad: hourly logrotate for dev nohup log (copytruncate, 10M x 3).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONF_SRC="${REPO_ROOT}/deploy/logrotate-rag-proxy-dev.conf"
CONF_DST="${HOME}/.config/rag-proxy/logrotate-dev.conf"
STATE="${HOME}/.config/rag-proxy/logrotate-dev.state"
CRON_MARK="# rag-proxy-dev-logrotate"

mkdir -p "$(dirname "${CONF_DST}")"
cp -f "${CONF_SRC}" "${CONF_DST}"

LOGROTATE="$(command -v logrotate)"
CRON_LINE="17 * * * * ${LOGROTATE} -s ${STATE} ${CONF_DST} >/dev/null 2>&1"
TMP="$(mktemp)"
(crontab -l 2>/dev/null | grep -v "${CRON_MARK}" || true) >"${TMP}"
echo "${CRON_MARK}" >>"${TMP}"
echo "${CRON_LINE} ${CRON_MARK}" >>"${TMP}"
crontab "${TMP}"
rm -f "${TMP}"

echo "[OK] installed ${CONF_DST}"
echo "[OK] crontab:"
crontab -l | grep "${CRON_MARK}" || true
