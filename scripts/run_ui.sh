#!/usr/bin/env bash
# Launch the Streamlit testing UI. Assumes the FastAPI backend is running
# (e.g. `make run` in another terminal).
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> No virtualenv. Run: make install"
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

CHATBOT_API_URL="${CHATBOT_API_URL:-http://127.0.0.1:8000}" \
exec streamlit run ui/streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --browser.gatherUsageStats false
