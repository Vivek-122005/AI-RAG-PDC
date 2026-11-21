# Agentic RAG with Self-Correction

Evidence-grounded question answering over a private document corpus. The agent decomposes
questions, retrieves evidence, drafts an answer, critiques it for groundedness and relevance,
and re-plans/re-retrieves when the critique fails — refusing honestly when the corpus can't
support an answer.

- [docs/PLAN.md](docs/PLAN.md): phased build plan
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): structure, layers and conventions

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com). Docker is optional.

```bash
cp .env.example .env
make install      # deps + pre-commit hooks
make check        # lint + typecheck + unit tests
make api          # http://localhost:8000/health
ollama pull qwen3:4b-instruct && ollama pull gemma3:4b
```

**Running on a laptop (8 GB).** By default Qdrant runs *embedded* in the Python process
(`QDRANT_PATH=data/index`), so no Docker VM is needed. Each embedding/reranker model uses
`ONNX_THREADS=2` and the agent retrieves one query at a time (`AGENT_MAX_CONCURRENCY=1`).
To use a Qdrant server instead, comment out `QDRANT_PATH` and run `make up`. Embedded mode
allows one process at a time, so don't run an eval while the API is serving.

## Ingest & search (Phase 1)

```bash
make ingest                                   # fetch 4 sample arXiv papers + index them
uv run rag ingest path/to/your/docs           # PDF, Markdown, HTML, TXT; re-runs skip unchanged files
uv run rag docs                               # list indexed documents
uv run rag search "how does CRAG grade retrieved documents?" --mode hybrid   # or dense / sparse
```

Ingestion keeps paragraph, page and heading structure, chunks within sections
(~350 tokens, 50-token overlap), prefixes each chunk with `title > section` for embedding,
and stores dense (bge-small) + BM25 sparse vectors in Qdrant for hybrid RRF search.

## Ask questions & evaluate (Phase 2)

Models run locally in Ollama (`qwen3:4b-instruct` answers, `gemma3:4b` grades):

```bash
ollama pull qwen3:4b-instruct && ollama pull gemma3:4b
uv run rag ask "What three actions can CRAG trigger?"      # answer with [n] citations
make eval-retrieval RUN=exp1                                # compare search modes / rerankers (no LLM)
make eval RUN=baseline                                      # answer → judge → eval/results/baseline/report.md
```

The golden set ([eval/datasets/golden.jsonl](eval/datasets/golden.jsonl)) has 45 questions:
single-hop, multi-hop, cross-paper comparisons and unanswerable ones the system must refuse.
Evidence is stored as verbatim quotes, so the set survives chunking changes.

## Self-correcting agent (Phase 3)

`rag ask` runs a LangGraph agent: plan → retrieve per sub-question → draft → critique →
re-plan and retrieve again (at most 2 retries) → answer or refuse.

```bash
uv run rag ask "Which model initializes CRAG's retrieval evaluator, and which model runs Ragas's prompts?" --trace
uv run rag ask --pipeline baseline "..."              # one-shot baseline, for comparison
make eval RUN=agent PIPELINE=agent                    # evaluate the agent
uv run rag eval all --run agent-no-critic --pipeline agent --no-critic   # ablations
uv run rag eval compare baseline-v1 agent-v1 agent-no-critic           # side-by-side table
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the graph and [docs/PLAN.md](docs/PLAN.md)
for how it compares with the baseline.

## Timeline

- **2026-11-11** — Repository created; project scoped and proposal written.
- **2026-11-14** — Implementation began (Phase 0).

## Status

- [x] Phase 0 — Foundations
- [x] Phase 1 — Ingestion & indexing
- [x] Phase 2 — Baseline RAG + evaluation
- [x] Phase 3 — Agentic orchestration
- [ ] Phase 4 — Guardrails
- [ ] Phase 5 — Observability
- [ ] Phase 6 — Serving
- [ ] Phase 7 — CI/CD & deployment
