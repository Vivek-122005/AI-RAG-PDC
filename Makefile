.PHONY: install up down lint format typecheck test test-integration api check samples ingest eval eval-retrieval

install:
	uv sync
	uv run pre-commit install

up:
	docker compose up -d --wait

down:
	docker compose down

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run lint-imports

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src tests

test:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration

api:
	uv run uvicorn agentic_rag.api.main:app --reload

check: lint typecheck test

samples:
	uv run python scripts/fetch_samples.py

ingest: samples
	uv run rag ingest corpus/samples

RUN ?= $(shell date +%Y%m%d-%H%M)

eval:  ## full baseline eval: make eval RUN=my-run
	uv run rag eval all --run $(RUN)

eval-retrieval:  ## retrieval-only ablation (no LLM)
	uv run rag eval retrieval --run $(RUN)
