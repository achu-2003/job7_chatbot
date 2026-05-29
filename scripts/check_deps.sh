#!/usr/bin/env bash
# Verifies the host has Postgres and Redis listening locally.
set -uo pipefail

ok=0
fail=0

check() {
  local name=$1 host=$2 port=$3
  if (echo > "/dev/tcp/${host}/${port}") 2>/dev/null; then
    printf '  \033[32mOK\033[0m   %-10s %s:%s\n' "$name" "$host" "$port"
    ok=$((ok+1))
  else
    printf '  \033[31mFAIL\033[0m %-10s %s:%s (not reachable)\n' "$name" "$host" "$port"
    fail=$((fail+1))
  fi
}

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

echo "==> Checking services"
check postgres "${POSTGRES_HOST:-127.0.0.1}" "${POSTGRES_PORT:-5432}"
check redis    "$(echo "${REDIS_URL:-redis://127.0.0.1:6379/0}" | sed -E 's#redis://([^:/]+).*#\1#')" \
               "$(echo "${REDIS_URL:-redis://127.0.0.1:6379/0}" | sed -E 's#.*:([0-9]+)/.*#\1#')"

echo
if [[ $fail -eq 0 ]]; then
  echo "All services reachable."
  exit 0
fi

cat <<MSG

Postgres is expected to be the remote SheScale DB — verify firewall / VPN
access if it failed. Install Redis locally with:
  sudo apt update && sudo apt install -y redis-server
  sudo systemctl enable --now redis-server
MSG
exit 1
