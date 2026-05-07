.PHONY: venv install install-dev test test-cov lint format format-check typecheck check clean

# ── Virtual environment ───────────────────────────────────────────────────────

PYTHON ?= python3
VENV   ?= .venv

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "  Created $(VENV)"

install-dev: venv
	. $(VENV)/bin/activate && pip install -e ".[dev]" -q
	@echo "  Installed relate[dev] in $(VENV)"

# ── Testing ───────────────────────────────────────────────────────────────────

test: install-dev
	. $(VENV)/bin/activate && python -m pytest tests/ -v --tb=short

test-cov: install-dev
	. $(VENV)/bin/activate && python -m pytest tests/ -v --tb=short --cov=relate --cov-report=term-missing

# ── Linting & formatting ─────────────────────────────────────────────────────

lint: install-dev
	. $(VENV)/bin/activate && python -m ruff check relate/ tests/

format: install-dev
	. $(VENV)/bin/activate && python -m ruff format relate/ tests/

format-check: install-dev
	. $(VENV)/bin/activate && python -m ruff format --check relate/ tests/

# ── Type checking ─────────────────────────────────────────────────────────────

typecheck: install-dev
	. $(VENV)/bin/activate && python -m mypy relate/ --ignore-missing-imports --check-untyped-defs

# ── Full check (lint + format + typecheck + test) ─────────────────────────────

check: lint format-check typecheck test

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf $(VENV)/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
