# Architecture

## Style: modular monolith

The system is **one deployable Python package** (`src/agentic_rag/`) split into modules
with strict, one-way dependencies.

**Why not microservices?** The work is a single request flow (retrieve → generate → critique),
run by a small team, on one machine during development. Splitting it into services would add
network hops, serialization, deployment and observability overhead without any scaling benefit.
The module boundaries below are drawn so that a piece can later be extracted into its own
service (for example ingestion workers) without rewriting it.

## Layers

Modules may import only from layers **below** them. Modules on the same row are siblings
and must not import each other.

```
┌──────────────────────────────────────────────────────────────┐
│  api   │   cli                        entrypoints (thin)     │
├──────────────────────────────────────────────────────────────┤
│  factory                              composition root       │
├──────────────────────────────────────────────────────────────┤
│  evaluation                           golden sets, metrics   │
├──────────────────────────────────────────────────────────────┤
│  pipelines                            baseline / agentic QA  │
├──────────────────────────────────────────────────────────────┤
│  ingestion   │   retrieval            write path │ read path │
├──────────────────────────────────────────────────────────────┤
│  indexing │ llm │ guardrails      adapters, rule-based checks │
├──────────────────────────────────────────────────────────────┤
│  domain                               shared data types      │
├──────────────────────────────────────────────────────────────┤
│  core                                 settings, logging      │
└──────────────────────────────────────────────────────────────┘
```

These rules are **enforced**, not aspirational: `import-linter` contracts in `pyproject.toml`
run in `make lint` and CI, and fail the build if an import breaks them. A second contract
forbids business logic (pipelines, retrieval, evaluation) from importing concrete providers
(`llm.ollama`, `indexing.fastembed`). Only `factory.py` and the entrypoints may do that.

## Module guide

| Module | Responsibility | Key files |
|---|---|---|
| `core/` | Settings from env/`.env` (12-factor), structured logging | `config.py`, `logging.py` |
| `domain/` | Plain data types shared by all layers: `Block`, `Document`, `Chunk`, `SearchHit`, stable IDs | `documents.py`, `search.py` |
| `llm/` | `LLM` Protocol (generate, structured output, stream) + Ollama adapter | `base.py`, `ollama.py` |
| `indexing/` | `Embedder` Protocol, fastembed adapter, Qdrant collection and hybrid search | `embedder.py`, `fastembed.py`, `store.py` |
| `ingestion/` | **Write path**: load files → structure-aware chunks → idempotent upsert | `loaders.py`, `chunking.py`, `pipeline.py` |
| `guardrails/` | Rule-based safety checks with no extra models: prompt-injection heuristics, PII redaction, input/output guards | `injection.py`, `pii.py`, `checks.py` |
| `retrieval/` | **Read path**: hybrid candidate search → cross-encoder rerank | `retriever.py`, `rerank.py` |
| `pipelines/` | Answer pipelines returning `Answer`: the one-shot baseline (control group) and the self-correcting agent | `baseline.py`, `answer.py`, `prompts.py` |
| `pipelines/guarded.py` | Wraps any pipeline with the input and output guards | `guarded.py` |
| `pipelines/agent/` | LangGraph agent: planner, per-query retrieval, drafter, critic, re-plan. Decision rules are pure functions | `graph.py`, `policy.py`, `evidence.py`, `schemas.py`, `prompts.py` |
| `evaluation/` | Golden datasets, synthesis, LLM judge, metrics, reports, ablations | `runner.py`, `metrics.py`, `report.py` |
| `factory.py` | Builds concrete objects from `Settings`. The only place that picks implementations | |
| `api/` | FastAPI app (health now, streaming query endpoints in Phase 6) | `main.py` |
| `cli/` | `rag` command; one module per command group, handlers stay thin | `main.py`, `index.py`, `query.py`, `evaluate.py` |

## Request flows

**Ingest** (`rag ingest <path>`)
```
cli.index → ingestion.pipeline
              ├─ loaders      file → Document(blocks with page + section path)
              ├─ chunking     Document → Chunks (section-bounded, overlapped)
              └─ indexing.store.upsert_document   (skips unchanged files by content hash)
                   └─ Embedder → dense + BM25 sparse vectors → Qdrant
```

**Answer, agent** (`rag ask "<question>"`, add `--trace` to see each step)
```
cli.query → factory.make_agent → pipelines.agent.AgenticRAG.answer   (LangGraph state machine)
  plan       LLM, structured: subjects named + self-contained sub-queries
  retrieve   one Retriever call per query, in parallel (LangGraph Send)
  gather     round-robin merge so every sub-question is represented; skip passages already shown
  draft      LLM: evidence-only answer with [n] citations (partial answers allowed)
  critique   LLM, structured: grounded? on target? what is missing? follow-up queries
             + deterministic guards (citations present, named subjects in cited passages)
  policy     accept │ refuse │ retry → replan: new queries + deeper ranks → retrieve …
```
The retry budget (`AgentConfig.max_retries`, default 2) bounds cost; an answer that is still
ungrounded or off-target when the budget runs out becomes the refusal.

**Answer, baseline** (`rag ask --pipeline baseline "<question>"`)
```
cli.query → factory.make_baseline → pipelines.BaselineRAG.answer
              ├─ retrieval.Retriever   Qdrant hybrid search (RRF) → reranker → top-k
              └─ llm.LLM.generate      evidence-only prompt with [n] citations → Answer
```

**Evaluate** (`rag eval all --run <name>`)
```
generate (system under test) → answers.jsonl
judge    (different model)   → judgments.jsonl
report                       → metrics.json + report.md
```

## Guardrails (defence in depth)

Every answer passes four independent layers, so one miss is not a breach:

| Layer | Where | What it stops |
|---|---|---|
| Input guard | `guardrails/checks.py` | empty or oversized input; direct prompt injection (weighted rules over NFKC-normalised text, zero-width characters removed); PII, which is redacted before the LLM or the logs see it |
| Untrusted documents | `ingestion/pipeline.py`, `Retriever(passage_filter=…)` | passages that carry instructions for the model: flagged at ingest, dropped at query time (so older indexes are covered too) |
| Spotlighting | `pipelines/prompts.py` | evidence is wrapped in `<evidence>` blocks it cannot close or forge; prompts say evidence is data |
| Output guard | `guardrails/checks.py` | citations to passages never shown are removed; answers left with no valid citation are refused; system-prompt leaks refused; PII redacted |

The detectors are regexes on purpose: a model-based classifier (e.g. Presidio, a DeBERTa
injection classifier) costs ~0.5 GB RAM each, which the 8 GB target can't spare. Each
sits behind one function (`detect_injection`, `find_pii`) so it can be swapped later.
On the 216 chunks of the sample corpus the injection rules flag nothing, and PII
detection finds only the authors' real e-mail addresses. `tests/redteam/` holds the
adversarial suite (direct and indirect injection, PII, fabricated citations, prompt
leaks, false positives); it runs with the unit tests in CI.

## Conventions

- **Depend on interfaces.** Cross-layer collaborators are `typing.Protocol`s (`LLM`, `Embedder`,
  `Reranker`) passed in through constructors. Tests use fakes (`tests/fakes.py`, the
  `FakeEmbedder` in `tests/conftest.py`), not mocks of vendor SDKs.
- **Configuration only through `Settings`.** No `os.environ` reads in business code.
- **Entrypoints are thin.** CLI and API handlers parse input, call one use case, format output.
  Heavy imports (ONNX models, clients) happen inside handlers so `rag --help` is instant.
- **LLMs propose, code decides.** Planner and critic return Pydantic-validated JSON; what
  happens next (accept, retry, refuse) is a pure function in `pipelines/agent/policy.py`,
  so the control flow is unit-tested without a model.
- **Prompts live next to their pipeline** (`pipelines/prompts.py`, `pipelines/agent/prompts.py`, `evaluation/judge.py`) as
  module constants, so prompt changes show up clearly in diffs and eval comparisons.
- **Data on disk is append-only JSONL** for evaluation runs, so long local runs can resume.
- **Typing:** `mypy --strict` on `src/` and `tests/`. **Style:** ruff (lint + format).

## Repository layout

```
.
├── src/agentic_rag/        the application package (layers above)
├── tests/
│   ├── unit/<module>/      mirrors src/; fast, no services (in-memory Qdrant, fakes)
│   ├── redteam/            adversarial guardrail cases (run with unit tests)
│   ├── integration/        needs `make up` + Ollama; auto-marked `integration`
│   ├── conftest.py         shared fixtures
│   └── fakes.py            test doubles for Protocols
├── eval/
│   ├── datasets/           golden sets (JSONL, committed)
│   └── results/            run outputs (git-ignored)
├── docs/                   PLAN.md (roadmap), ARCHITECTURE.md (this file)
├── scripts/                one-off developer utilities (sample corpus download)
├── corpus/                 local documents (git-ignored)
├── compose.yaml            local infrastructure (Qdrant)
├── Makefile                developer commands (`make help`-style targets)
└── .github/workflows/      CI: lint, import contracts, types, unit + integration tests
```

## Adding something new

1. Decide its layer. If it's a new data type used across modules, put it in `domain/`.
2. If it talks to an external system, define a Protocol where it's consumed, put the adapter
   in the infrastructure layer, and construct it only in `factory.py`.
3. Add unit tests under `tests/unit/<module>/` using fakes.
4. Run `make check`. Import contracts will tell you if a dependency points the wrong way.
5. Update this document if you added a module.
