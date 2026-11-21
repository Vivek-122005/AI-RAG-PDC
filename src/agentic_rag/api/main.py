"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import anyio
import httpx
import structlog
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from agentic_rag import __version__
from agentic_rag.core.config import Settings, get_settings
from agentic_rag.core.logging import configure_logging
from agentic_rag.core.telemetry import configure_from_settings

log = structlog.get_logger()


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    qdrant: Literal["up", "down"]
    ollama: Literal["up", "down"]
    models_missing: list[str]


async def qdrant_is_up(settings: Settings) -> bool:
    if settings.qdrant_path:  # embedded: opening it here would fight the owner's file lock
        return await anyio.Path(settings.qdrant_path).is_dir()
    client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=2)
    try:
        await client.get_collections()
        return True
    except Exception as exc:
        log.warning("qdrant_unreachable", url=settings.qdrant_url, error=str(exc))
        return False
    finally:
        await client.close()


async def ollama_models(settings: Settings) -> set[str] | None:
    """Return the model tags installed in Ollama, or None if Ollama is unreachable."""
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=2) as client:
            r = await client.get("/api/tags")
            r.raise_for_status()
            return {m["name"] for m in r.json().get("models", [])}
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("ollama_unreachable", url=settings.ollama_base_url, error=str(exc))
        return None


def missing_models(settings: Settings, installed: set[str]) -> list[str]:
    wanted = {settings.generator_model, settings.critic_model}
    # Ollama reports "name:latest" when no tag was given.
    normalized = {m.removesuffix(":latest") for m in installed} | installed
    return sorted(m for m in wanted if m not in normalized)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    configure_from_settings(settings)
    log.info("startup", env=settings.app_env, version=__version__)
    yield


app = FastAPI(title="Agentic RAG", version=__version__, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    q_up = await qdrant_is_up(settings)
    installed = await ollama_models(settings)
    o_up = installed is not None
    missing = missing_models(settings, installed) if installed is not None else []
    return HealthResponse(
        status="ok" if q_up and o_up and not missing else "degraded",
        version=__version__,
        qdrant="up" if q_up else "down",
        ollama="up" if o_up else "down",
        models_missing=missing,
    )
