# Agentic RAG with Self-Correction

Evidence-grounded question answering over a private document corpus. The agent decomposes
questions, retrieves evidence, drafts an answer, critiques it for groundedness and relevance,
and re-plans/re-retrieves when the critique fails — refusing honestly when the corpus can't
support an answer.

See [docs/PLAN.md](docs/PLAN.md) for the phased build plan.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
cp .env.example .env
make install      # deps + pre-commit hooks
make up           # Qdrant
make check        # lint + typecheck + unit tests
make api          # http://localhost:8000/health
```

## Timeline

- **2026-11-11** — Repository created; project scoped and proposal written.
- **2026-11-14** — Implementation began (Phase 0).

## Status

- [x] Phase 0 — Foundations
- [ ] Phase 1 — Ingestion & indexing
- [ ] Phase 2 — Baseline RAG + evaluation
- [ ] Phase 3 — Agentic orchestration
- [ ] Phase 4 — Guardrails
- [ ] Phase 5 — Observability
- [ ] Phase 6 — Serving
- [ ] Phase 7 — CI/CD & deployment
