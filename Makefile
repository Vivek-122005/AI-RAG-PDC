.PHONY: install up down traces lint format typecheck test test-integration api check samples ingest eval eval-retrieval eval-gate image serve loadtest

install:
	uv sync
	uv run pre-commit install

up:
	docker compose up -d --wait

traces:  ## optional Jaeger trace viewer on http://localhost:16686 (set TRACING=otlp)
	docker compose --profile observability up -d jaeger

down:
	docker compose --profile observability --profile app down

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
	uv run pytest tests/unit tests/redteam

test-integration:
	uv run pytest tests/integration

api:
	uv run uvicorn agentic_rag.api.main:app --reload

check: lint typecheck test

image:  ## production image (multi-stage, non-root, models baked in)
	docker build -t agentic-rag:local .

serve:  ## Qdrant + API in containers on :8000 (needs API_KEYS; Ollama on the host)
	docker compose --profile app up -d --build --wait

loadtest:  ## light load test against a running API: make loadtest [N=20 C=2 URL=...]
	uv run python scripts/loadtest.py --requests $(or $(N),20) --concurrency $(or $(C),2) --url $(or $(URL),http://localhost:8000) --key "$(KEY)"

samples:
	uv run python scripts/fetch_samples.py

ingest: samples
	uv run rag ingest corpus/samples

PIPELINE ?= baseline
RUN ?= $(shell date +%Y%m%d-%H%M)

eval:  ## full eval: make eval RUN=my-run [PIPELINE=agent]
	uv run rag eval all --run $(RUN) --pipeline $(PIPELINE)

eval-retrieval:  ## retrieval-only ablation (no LLM)
	uv run rag eval retrieval --run $(RUN)

eval-gate:  ## CI quality gate: retrieval metrics vs floors (no LLM)
	uv run rag eval gate
