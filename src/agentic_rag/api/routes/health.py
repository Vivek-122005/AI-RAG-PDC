"""GET /health: liveness plus the state of the index and the model server."""

import anyio
import httpx
import structlog
from fastapi import APIRouter

from agentic_rag import __version__
from agentic_rag.api.schemas import Health
from agentic_rag.api.state import StateDep
from agentic_rag.core.config import Settings

router = APIRouter(tags=["health"])
log = structlog.get_logger()


async def ollama_models(settings: Settings) -> set[str] | None:
    """Model tags installed in Ollama, or None if Ollama is unreachable."""
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
    normalized = {m.removesuffix(":latest") for m in installed} | installed
    return sorted(m for m in wanted if m not in normalized)


@router.get("/health")
async def health(state: StateDep) -> Health:
    try:
        with anyio.fail_after(3):
            await anyio.to_thread.run_sync(state.services.store.count)
        index_up = True
    except Exception as exc:  # any failure means "down"; the reason goes to the log
        log.warning("index_unreachable", error=str(exc))
        index_up = False
    installed = await ollama_models(state.settings)
    missing = missing_models(state.settings, installed) if installed is not None else []
    healthy = index_up and installed is not None and not missing
    return Health(
        status="ok" if healthy else "degraded",
        version=__version__,
        index="up" if index_up else "down",
        ollama="up" if installed is not None else "down",
        models_missing=missing,
    )
