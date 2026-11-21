# Deployment

The service is one container (API + `rag` CLI) plus Qdrant. Ollama runs outside the
container: on a Mac it must stay on the host to use the GPU; on a Linux server it can be
its own container or a GPU node.

## Image

`Dockerfile` is multi-stage:

| Stage | Purpose |
|---|---|
| `build` | `uv sync --locked --no-dev` into `/app/.venv`; dependency layer cached until `uv.lock` changes |
| `models` | downloads the dense, BM25 and reranker models into `/models` at build time, so startup does no network I/O |
| `runtime` | `python:3.12-slim`, no build tools, non-root user (uid 10001), `/data` volume, `HEALTHCHECK` on `/health` |

Defaults baked in: `APP_ENV=prod` (refuses to start without `API_KEYS`), JSON logs,
one uvicorn worker (the models are loaded once per process and the LLM is the bottleneck;
scale with replicas, not workers).

```bash
make image                                       # docker build -t agentic-rag:local .
```

## Local, in containers

```bash
export API_KEYS="demo-key:demo"
make serve                                       # Qdrant + API on :8000, Ollama on the host
docker compose run --rm api rag ingest /corpus/samples   # index the shared corpus once
make loadtest KEY=demo-key                       # p50/p95 latency, status mix, cache hits
make down
```

Memory limits in `compose.yaml`: API 1.5 GB, Qdrant 512 MB, Jaeger 256 MB. With the 4B
models in Ollama (~3 GB each, loaded one at a time) this fits an 8 GB laptop.

## Configuration

All settings are environment variables (see `.env.example` and `core/config.py`). The ones
that matter in production:

| Variable | Notes |
|---|---|
| `API_KEYS` | `key:tenant,...`. Required when `APP_ENV=prod`. Inject from a secret store, never bake into the image |
| `QDRANT_URL` / `QDRANT_API_KEY` | a Qdrant server or Qdrant Cloud. `QDRANT_PATH` (embedded) is single-process: dev and tests only |
| `OLLAMA_BASE_URL` | where the models are served; `GENERATOR_MODEL`, `CRITIC_MODEL` must be pulled there |
| `API_MAX_CONCURRENT_JOBS` | answers/ingests run at once per replica; raise only with a GPU that can serve them |
| `API_RATE_LIMIT_PER_MINUTE` | per tenant, per replica (in-memory). Put a shared limiter at the gateway for many replicas |
| `TRACING=otlp`, `OTLP_ENDPOINT` | any OpenTelemetry collector (Jaeger, Tempo, Honeycomb, ...) |
| `CORS_ORIGINS` | only if a browser app on another origin calls the API |

## CI/CD

| Workflow | When | What |
|---|---|---|
| `ci.yml` → `checks` | every push and PR | ruff, import contracts, mypy --strict, unit + red-team tests |
| `ci.yml` → `integration` | every push and PR | tests against a Qdrant service container |
| `ci.yml` → `eval-gate` | after `checks` | downloads the sample corpus, indexes it, runs `rag eval gate`: fails if hit_rate@5 or evidence_recall@5 drops below 0.80 (currently 0.886 / 0.871). Metrics uploaded as an artifact |
| `image.yml` | PRs touching the image; pushes to `main`; `v*` tags | builds with layer cache, smoke-tests `/health` in the container; on `main`/tags pushes `ghcr.io/<owner>/<repo>` tagged by branch, semver and sha |

The gate covers retrieval only because it is deterministic and needs no LLM. Answer
quality (judge-scored correctness, faithfulness, refusals) needs Ollama and about an hour
on a laptop, so it is a local release check: `make eval RUN=<name> PIPELINE=agent`, then
`rag eval compare <previous> <name>`.

## Going further (not done here)

- Several replicas need a shared cache and rate limiter (Redis) and a queue for ingest
  jobs instead of in-process background tasks; job status is in memory today.
- Uploads live on the `/data` volume; use object storage for more than one replica.
- Serve the LLM with vLLM or TGI on a GPU for real concurrency; the `LLM` Protocol keeps
  that a one-adapter change.
