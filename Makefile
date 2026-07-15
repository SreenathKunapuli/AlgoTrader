PY := .venv/bin/python
PIP := .venv/bin/pip

# Port the API server binds to. Override locally in an untracked Makefile.local.
API_PORT ?= 8000
-include Makefile.local

.PHONY: setup test lint typecheck run-engine run-api run-web run-all train-signal evaluate-signal

setup:
	test -d .venv || python3 -m venv .venv
	$(PIP) install -q -e ".[dev]"

test:
	$(PY) -m pytest engine/tests api/tests -q

test-research:
	cd research && ../$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check engine api

typecheck:
	$(PY) -m mypy

run-engine:
	# caffeinate: a sleeping laptop is a dead trading engine (macOS)
	caffeinate -is $(PY) -m lobplatform.cli run --tier medium

run-api:
	$(PY) -m uvicorn app.main:app --app-dir api --port $(API_PORT)

run-web:
	cd web && npm run dev

run-all:
	@echo "Run in 3 terminals: make run-engine | make run-api | make run-web"

train-signal:
	$(PY) -m lobplatform.cli train-signal

evaluate-signal:
	$(PY) -m lobplatform.cli evaluate-signal
