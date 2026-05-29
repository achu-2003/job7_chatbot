#!/usr/bin/env bash
# Triggers a full re-embed of products/faqs/policies.
set -euo pipefail

if [[ -f .env ]]; then set -a; source .env; set +a; fi
HOST="${APP_HOST:-127.0.0.1}"
PORT="${APP_PORT:-8000}"

curl -fsS -X POST "http://${HOST}:${PORT}/api/v1/admin/reindex" \
     -H 'content-type: application/json' \
     -d '{}' | python -m json.tool
