# Agentic RAG with Self-Correction — Build Plan

> Living document. Each phase ends in something **runnable, tested, and measured** — and the next phase builds on it.
> Last updated: 2026-11-14

---

## 1. What we are building (restated)

A question-answering service over a **private document corpus** that does not trust its first answer.

```
            ┌──────────────┐
 question → │ input guard  │──(injection / PII / off-topic)──► refuse
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │   planner    │  classify + decompose into sub-questions
            └──────┬───────┘
                   ▼  (fan-out, one per sub-question)
            ┌──────────────┐
            │  retriever   │  hybrid (dense + BM25) → rerank → grade relevance
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │   drafter    │  answer with inline citations [doc:chunk]
            └──────┬───────┘
                   ▼
            ┌──────────────┐   groundedness < τ  or  relevance < τ
            │    critic    │──────────────────────────────┐
            └──────┬───────┘                              ▼
                   │ pass                          ┌────────────┐
                   ▼                               │  re-planner │ rewrite queries,
            ┌──────────────┐                       │ (≤ N loops) │ widen/narrow search
            │ output guard │                       └─────┬──────┘
            └──────┬───────┘                             │
                   ▼                                     └──► back to retriever
        answer + citations + confidence      (loop budget exhausted → honest refusal)
```

**Core promise to the user:** every claim is traceable to a chunk, and when the corpus doesn't support an answer, the system says so instead of guessing.

**Target users:** students, researchers, professionals with large private document sets.

---

## 2. Research-backed stack decisions

| Concern | Choice | Why (and what I rejected) |
|---|---|---|
| Language / env | **Python 3.12** via `uv` | Your machine has 3.14, but ONNX/torch-based libs (fastembed, rerankers, presidio) lag new Pythons. 3.12 is the safe production target. |
| Orchestration | **LangGraph 1.2.x** | Explicit state machine, conditional edges, `Send` for parallel fan-out, checkpointers, first-class streaming. Exactly the "loop until critic passes" shape. |
| Vector DB | **Qdrant** (not pgvector) | Native hybrid search (dense + sparse/BM25) with RRF/DBSF fusion in a single Query API call, payload filtering, runs as one container. pgvector needs extra work for hybrid. |
| Embeddings | **fastembed**: `BAAI/bge-small-en-v1.5` (dense) + `Qdrant/bm25` (sparse); reranker `BAAI/bge-reranker-base` | Open-weight, ONNX, CPU-friendly, no GPU needed. Swap to `bge-m3` later if multilingual. |
| Document parsing | **PyMuPDF** (fast) → optionally **Docling** for tables/layout | Start simple; upgrade only if eval shows table questions failing. |
| LLM | **Ollama, local** (decided 2026-11-14), behind a provider-agnostic interface | Private, free, offline. Ollama supports JSON-schema structured output (needed for planner/critic). Run Ollama **natively** on macOS, not in Docker (Docker on Mac has no Metal GPU). See §2.1 for model sizing. |
| Evaluation | **Ragas** (faithfulness, answer relevancy, context precision/recall) + own retrieval metrics (recall@k, MRR) | Ragas covers generation quality; ID-based retrieval metrics are free and deterministic. |
| Guardrails | **Layered, mostly custom** + **Presidio** (PII) + a prompt-injection classifier (e.g. `protectai/deberta-v3-base-prompt-injection-v2`) | NeMo Guardrails / Guardrails AI are heavy and opaque for learning. Building the layers yourself teaches more; you can still wrap Guardrails AI validators later. |
| Observability | **OpenTelemetry + OpenInference → Arize Phoenix** | Phoenix is one container. Self-hosted Langfuse v3 needs Postgres + ClickHouse + Redis + S3 — too much infra for this stage. Since we emit OTel, switching later is cheap. |
| Cache / rate limit | **Redis** | Embedding + answer caching, rate limiting, optionally LangGraph checkpoints. Introduced in Phase 6, not before it's needed. |
| API | **FastAPI** + SSE streaming | Stream node-level progress ("retrieving…", "critic rejected, re-planning…") plus answer tokens. |
| Packaging / CI | Docker Compose, GitHub Actions, GHCR | Standard. Cloud target chosen in Phase 7. |
| Quality tooling | ruff, mypy, pytest, pre-commit | From day one. |

### 2.1 Local model sizing (M2, 8 GB RAM)

8 GB is shared between macOS, Qdrant, fastembed models, the API and the LLM, so **~4 GB is the realistic LLM budget**.

| Role | Default | Notes |
|---|---|---|
| Generator / drafter | `qwen3:4b-instruct` (~2.5 GB, Q4) | Good instruction-following and JSON output for its size. **Not** `qwen3:4b`: that tag is the thinking-only model, which always emits long reasoning (slow, and `think=false` leaks it into content). Alternative: `llama3.2:3b`. |
| Planner / critic / grader | same model, different prompts | Loading one model avoids swapping. Keep `think` off for speed except where it measurably helps the critic. |
| Eval judge (offline only) | `gemma3:4b` (~3.4 GB) | Different model family from the generator, which reduces self-grading bias. `gemma2:9b` was tried first: at 8k context it needs 7.3 GB and spills half onto CPU on an 8 GB machine, making evals impractically slow. |
| Question synthesis | `qwen3:4b-instruct` | Drafts only; every item is reviewed by hand. |

**Risk:** small models are weaker critics. Phase 3 must measure critic precision/recall against hand labels; if the critic is unreliable, options are a reranker-score/NLI-based groundedness check (cheap, deterministic) combined with the LLM verdict, or a hosted judge for eval only. Keep context windows modest (`num_ctx` 8k) — retrieved evidence must fit.

---

## 3. The phases

Each phase lists **parts**, an **exit criterion**, and **what you learn**. Don't start a phase until the previous exit criterion is met.

### Phase 0 — Foundations (repo, tooling, infra skeleton)
1. `uv` project, `src/` layout, pinned Python 3.12.
2. Config via `pydantic-settings` (`.env`), structured logging.
3. Docker Compose with Qdrant; `make up / down / test / lint`.
4. FastAPI app with `/health` (checks Qdrant connectivity).
5. ruff + mypy + pytest + pre-commit; GitHub Actions running lint + tests.
6. **Start the golden eval set**: pick your corpus (10–50 real docs you care about) and write 20 Q/A pairs by hand, including 5 "not answerable from corpus" questions.

**Exit:** `make up && make test` green locally and in CI.
**Learn:** project hygiene, reproducible envs, 12-factor config.

### Phase 1 — Ingestion & Indexing
1. Loaders: PDF, Markdown, TXT, HTML (DOCX optional). Normalize to a `Document` model.
2. Chunking: structure-aware recursive splitter (headings → paragraphs → sentences), target ~400–600 tokens, ~10–15% overlap. Semantic chunking as an experiment, not default.
3. Metadata: `doc_id`, `source`, `title`, `page`, `section_path`, `chunk_index`, `content_hash`, `ingested_at`, user tags.
4. Idempotency: content hashing → re-ingesting an unchanged file is a no-op; changed file replaces its chunks.
5. Qdrant collection with named vectors `dense` + `sparse`, payload indexes on `doc_id`/`source`.
6. CLI: `rag ingest ./corpus`, `rag search "query"` (raw hybrid results).

**Exit:** corpus ingested; manual `rag search` returns sensible chunks; unit tests for chunker and idempotency.
**Learn:** chunking trade-offs, hybrid retrieval, vector DB ops.

### Phase 2 — Baseline RAG + Evaluation Harness  ⚠️ *most important phase*
1. LLM abstraction (`generate`, `generate_structured`, streaming) with an Ollama implementation.
2. Naive pipeline: hybrid retrieve top-k → rerank → generate with forced citations.
3. Expand golden set to 50–100 questions (use LLM-generated synthetic questions, then **hand-verify**). Tag each: single-hop, multi-hop, comparison, unanswerable.
4. Eval runner: retrieval metrics (recall@k, MRR via chunk IDs) + Ragas (faithfulness, answer relevancy, context precision/recall) + refusal accuracy on unanswerables.
5. Results written to `eval/results/<timestamp>.json` + a markdown report.
6. Retrieval tuning experiments: k, chunk size, with/without reranker, dense-only vs hybrid. Record the table.

**Exit:** a baseline scorecard you trust. *Without this, you cannot prove self-correction helps.*
**Learn:** LLM evaluation methodology, LLM-as-judge caveats, retrieval tuning with data.

#### Phase 2 results (run `baseline-v1`, 2026-11-14)

Setup: 45-item golden set (25 single-hop, 4 multi-hop, 6 comparison, 10 unanswerable),
hybrid RRF search (20 candidates) → MiniLM-L12 rerank → top-5, `qwen3:4b-instruct` answers,
`gemma3:4b` judges. Ragas itself was replaced by our own claim-level faithfulness and 1–5
correctness judge, because it is easier to run offline on 8 GB and to inspect.

**Retrieval ablation** (35 answerable items, no LLM):

| config | hit@5 | hit@1 | MRR@5 | evidence recall@5 |
|---|---|---|---|---|
| dense / none | 0.686 | 0.200 | 0.390 | 0.629 |
| sparse / none | **0.914** | 0.571 | 0.691 | 0.829 |
| hybrid / none | 0.829 | 0.457 | 0.601 | 0.771 |
| dense / MiniLM-L12 | 0.800 | 0.543 | 0.645 | 0.757 |
| **hybrid / MiniLM-L12** (default) | 0.886 | **0.600** | **0.700** | **0.871** |
| hybrid / jina-tiny | 0.857 | 0.486 | 0.618 | 0.814 |

Caveat: BM25 is flattered because synthesized questions reuse passage wording. Hand-written
questions should narrow the sparse/dense gap.

**Baseline scorecard:**

| | overall | single-hop | multi-hop | comparison | unanswerable |
|---|---|---|---|---|---|
| retrieval hit@5 | 0.886 | 0.960 | 1.000 | **0.500** | - |
| evidence recall@5 | 0.871 | 0.960 | 1.000 | **0.417** | - |
| accuracy (correctness ≥ 4) | 0.886 | 0.960 | 1.000 | **0.500** | - |
| false refusal rate | 0.114 | 0.040 | 0.000 | **0.500** | - |
| correct refusal rate | - | - | - | - | 0.900 |
| latency p50 / p95 (s) | 39 / 73 | | | | |

What this tells Phase 3:
- **Comparison questions fail at retrieval, not generation.** One query cannot pull evidence
  from two papers into the top-5, so the model honestly refuses (3 of 5 failures). This is
  exactly what question decomposition is for.
- **Entity substitution:** for "What GPU did the *Ragas* authors use…" it answered with the
  *RAG* paper's V100s, citing it. The critic needs a question-relevance check, not only
  "is each claim supported by the evidence".
- **The judge does not discriminate.** gemma3:4b gave every non-refused answer 5/5 and full
  faithfulness, including one answer whose reasoning contradicts itself (mh-06). The judge
  scores are a ceiling; until the judge is calibrated against a hand-labelled sample, trust
  the deterministic metrics (retrieval, refusals, cited evidence) more than the judge scores.

### Phase 3 — Agentic Orchestration (the core)
1. LangGraph `State` (question, sub_questions, evidence, draft, critique, iteration, trace).
2. **Planner**: classify (simple / multi-hop / out-of-scope); decompose multi-hop into sub-questions (structured output).
3. **Retriever node**: parallel per sub-question via `Send`; LLM or reranker-score relevance grading to drop junk.
4. **Drafter**: answer from graded evidence only, with citations.
5. **Critic**: structured verdict — claim-level groundedness (each sentence supported by a cited chunk?), answer relevance, completeness; returns *what is missing*.
6. **Re-planner**: uses the critique to rewrite/expand queries; loop budget (e.g. max 2 retries) and refusal path.
7. Re-run the Phase 2 eval: compare baseline vs agentic on quality, latency, tokens. Also ablations (critic off, decomposition off).

**Exit:** agentic beats baseline on multi-hop + unanswerable categories, with known latency/cost overhead.
**Learn:** agentic control flow, structured outputs, state machines, honest ablation.

### Phase 4 — Guardrails & Safety
1. **Input**: length/format limits; prompt-injection classifier + heuristics; PII detection/redaction (Presidio) on queries.
2. **Indirect injection** (documents are untrusted!): sanitize retrieved text, wrap in delimited "data" blocks (spotlighting), instruct the model never to follow instructions inside evidence; scan at ingest time and flag suspicious chunks.
3. **Output**: citation validator (every cited ID exists and was retrieved), groundedness threshold → refuse or hedge, toxicity check, PII leak check.
4. Red-team suite: ~30 adversarial cases (direct injection, poisoned document, jailbreaks, PII extraction) as automated tests.

**Exit:** red-team suite passes; false-refusal rate on golden set measured and acceptable.
**Learn:** threat modelling for LLM apps, defence-in-depth, safety vs helpfulness trade-off.

### Phase 5 — Observability
1. OpenTelemetry + OpenInference instrumentation for LangGraph/LLM calls; Phoenix in Compose.
2. Per-request trace: every node span with latency, tokens, cost, model.
3. Custom attributes: critic verdict, loop count, groundedness score, refusal reason.
4. Failure taxonomy dashboard: retrieval miss / critic false-reject / loop exhaustion / guardrail block.
5. Structured JSON logs correlated by trace ID.

**Exit:** you can open any bad answer and see exactly which node went wrong.
**Learn:** LLM-specific tracing, production diagnostics.

### Phase 6 — Serving (FastAPI)
1. Endpoints: `POST /v1/query` (SSE stream), `POST /v1/documents` (upload → background ingest), `GET /v1/documents`, `DELETE /v1/documents/{id}`.
2. Stream events: `status` (node progress), `token`, `citations`, `final`.
3. Redis: answer cache, embedding cache, rate limiting.
4. API-key auth; per-collection (per-user) isolation via Qdrant payload filter.
5. Integration tests with `httpx` + Testcontainers (or compose in CI).
6. Optional thin UI (Streamlit or a small web page) for demos.

**Exit:** end-to-end streaming answer through the API in Docker.
**Learn:** async API design, streaming, multi-tenant data isolation.

**As built (2026-11-21).** All endpoints, API-key → tenant auth, tenant payload filter,
per-tenant rate limit and answer cache, demo page at `/`. Changed from the plan:
- No `token` events: the critic can still reject a draft, so streaming its tokens would
  show text that is then withdrawn. Clients get `status` per node, then `citations`, `final`.
- No Redis yet: cache and rate limiter are in-process (one replica on a laptop). They sit
  behind small classes (`api/cache.py`, `api/security.py`) so Redis is a drop-in later.
- API tests run the real app in-process (httpx ASGI transport, in-memory Qdrant, fake
  LLM) instead of Testcontainers, so they run with the unit suite in seconds.

### Phase 7 — CI/CD, Eval Gates & Deployment
1. Multi-stage Dockerfile (slim, non-root, models pre-downloaded).
2. Full Compose stack: api + qdrant + redis + phoenix.
3. GitHub Actions: lint → unit → integration → **eval gate** (small eval subset; fail PR if faithfulness/recall drop below thresholds) → build & push image to GHCR.
4. Deploy: pick one — a single VM (simplest, Compose as-is), Fly.io/Render, or Cloud Run + Qdrant Cloud.
5. Secrets management, health checks, basic load test (locust/k6) to find latency limits.

**Exit:** a public (auth-protected) URL, and a PR that degrades quality gets blocked automatically.
**Learn:** operationalizing ML systems, regression testing for LLMs.

**As built (2026-11-21).** Multi-stage non-root image with models baked in; Compose
`app` profile (api + qdrant, Jaeger optional); CI eval gate; image build, smoke test and
GHCR push; `scripts/loadtest.py`; [DEPLOY.md](DEPLOY.md). Changed from the plan:
- The eval gate checks retrieval (hit_rate@5, evidence_recall@5 ≥ 0.80) rather than
  faithfulness: retrieval is deterministic and LLM-free, so it can block a PR on a free
  runner. Judge-scored answer quality remains a local release check.
- No public URL yet: hosting a 4B model needs a GPU host, which is a cost decision. The
  image runs anywhere containers run (single VM with Compose is the simplest target).
- Load test is a small httpx script rather than locust/k6 (no extra dependency).

---

## 4. Review: complexity & usefulness

**Complexity: high, but well-shaped.** Roughly 8–12 weeks part-time. The hard parts are *not* the frameworks — they're:

1. **Evaluation honesty.** The critic is itself an LLM; it can be wrong both ways. You must measure critic precision/recall against human labels on ~50 cases, otherwise the "self-correction" claim is unproven.
2. **Latency & cost.** Each correction loop roughly doubles LLM calls. Mitigate with: loop budget, skipping decomposition for simple questions (router), cheaper model for critic/grader, caching.
3. **Indirect prompt injection** via documents is the realistic threat, not users typing "ignore previous instructions".
4. **Scope creep.** The 7-stage list is a lot. Phases 0–3 are the real project; 4–7 make it production-grade.

**Usefulness: strong.** It's a real problem (hallucination over private docs), and the project exercises nearly every skill asked of an AI/LLM engineer: retrieval, agents, eval, safety, observability, serving, MLOps. The differentiator in a portfolio is **the eval numbers**: "agentic + critic improved faithfulness from X→Y and correct-refusal from A→B at +Z ms latency" beats any feature list.

**Suggestions vs the original proposal:**
- Swap Langfuse → Phoenix initially (lighter); keep OTel so either works.
- Drop NeMo Guardrails initially; build layered guardrails yourself + Presidio.
- Pick Qdrant over pgvector (hybrid out of the box).
- Add Phase 2 (baseline + eval) explicitly — it was missing and it's what makes the rest credible.
- Add a router so simple questions skip the expensive path.

**Open decisions:**
- ~~LLM provider~~ → Ollama, local (see §2.1).
- The corpus you'll use for development and evaluation.
- Cloud target (Phase 7).

---

## 5. Repo layout

See [ARCHITECTURE.md](ARCHITECTURE.md): modular monolith with enforced layers (`core → domain → indexing/llm → ingestion/retrieval → pipelines → evaluation → factory → api/cli`).

## 6. Working agreement

- **Commits are made only by the repo owner (`vivek-122005`)** — Claude never commits or pushes.
- Each part ends with tests passing; each phase ends with its exit criterion met.
