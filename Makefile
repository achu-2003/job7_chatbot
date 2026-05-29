# ============================================================
# Native (no Docker) workflow
# ============================================================
PYTHON  ?= /usr/bin/python3.12
VENV    ?= .venv
PIP     := $(VENV)/bin/pip
PYBIN   := $(VENV)/bin/python

.PHONY: help install env check run ui reindex test clean

help:
	@echo "make install     - create venv + install requirements"
	@echo "make env         - copy .env.native -> .env (if missing)"
	@echo "make check       - verify Postgres (remote) + Redis (local) reachable"
	@echo "make run         - start FastAPI (with bundled worker, embedded Chroma)"
	@echo "make ui          - start Streamlit testing UI (talks to make run)"
	@echo "make reindex     - enqueue full re-embed (products from live DB + FAQs/policies)"
	@echo "make test        - run unit tests"
	@echo "make clean       - remove venv and local chroma data"

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements.txt
	$(PIP) install uvloop httptools

install: $(VENV)/bin/activate

env:
	@if [ ! -f .env ]; then \
	  cp .env.native .env; \
	  echo "Created .env from .env.native. Edit it to set OPENAI_API_KEY."; \
	else \
	  echo ".env already exists."; \
	fi

check:
	bash scripts/check_deps.sh

run:
	bash scripts/run.sh

ui:
	bash scripts/run_ui.sh

reindex:
	bash scripts/reindex.sh

test: install
	$(PYBIN) -m pytest -q

clean:
	rm -rf $(VENV) data/chroma __pycache__ .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
