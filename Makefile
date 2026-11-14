.PHONY: install up down lint format typecheck test test-integration api check samples ingest

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

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src

test:
	uv run pytest -m "not integration"

test-integration:
	uv run pytest -m integration

api:
	uv run uvicorn agentic_rag.api.main:app --reload

check: lint typecheck test

samples:
	uv run python scripts/fetch_samples.py

ingest: samples
	uv run rag ingest corpus/samples
