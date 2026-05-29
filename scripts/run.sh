#!/usr/bin/env bash
# Native run: starts the FastAPI app (with embedded ChromaDB and bundled
# embedding worker). Requires Postgres + Redis already running locally.
set -euo pipefail

if [[ ! -f .env ]]; then
  echo "No .env found. Copy .env.native -> .env and set OPENAI_API_KEY first."
  exit 1
fi

set -a; source .env; set +a

VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> No virtualenv. Run: make install"
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

HOST="${APP_HOST:-127.0.0.1}"
PORT="${APP_PORT:-8000}"

mkdir -p "${CHROMA_PERSIST_DIR:-./data/chroma}"

echo "==> uvicorn ${HOST}:${PORT} (vector_mode=${VECTOR_MODE:-http}, worker_in_api=${RUN_WORKER_IN_API:-false})"
exec uvicorn app.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1 \
  --loop uvloop \
  --http httptools
