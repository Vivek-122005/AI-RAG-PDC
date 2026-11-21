# syntax=docker/dockerfile:1
# Production image for the API (and the `rag` CLI). Ollama is not inside: it runs on the
# host (Apple GPU) or as its own service, reached through OLLAMA_BASE_URL.

ARG PYTHON=3.12

# ── 1. dependencies and the app, into /app/.venv ──────────────────────────────────────
FROM python:${PYTHON}-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first: this layer is reused until uv.lock changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable

# ── 2. embedding, BM25 and reranker models, baked in (no download at startup) ─────────
FROM build AS models
ENV FASTEMBED_CACHE_PATH=/models
RUN /app/.venv/bin/python -c "\
from agentic_rag.core.config import Settings; \
from fastembed import SparseTextEmbedding, TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
s = Settings(); \
TextEmbedding(s.dense_model); SparseTextEmbedding(s.sparse_model); TextCrossEncoder(s.reranker_model)"

# ── 3. runtime: slim, non-root, no build tools ────────────────────────────────────────
FROM python:${PYTHON}-slim AS runtime
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --from=models --chown=app:app /models /models
RUN mkdir -p /data && chown app:app /data
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models \
    APP_ENV=prod \
    LOG_JSON=true \
    UPLOAD_DIR=/data/uploads
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
# One worker: models are loaded once per process, and the laptop LLM is the bottleneck.
CMD ["uvicorn", "agentic_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
