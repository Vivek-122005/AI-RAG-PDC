import pytest
from httpx import ASGITransport, AsyncClient

from agentic_rag.api import main
from agentic_rag.config import Settings, get_settings


@pytest.fixture
def client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=main.app), base_url="http://test")


async def test_health_degraded_when_qdrant_down(client: AsyncClient) -> None:
    # Point at a port where nothing listens.
    main.app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, qdrant_url="http://127.0.0.1:1", ollama_base_url="http://127.0.0.1:1"
    )
    try:
        r = await client.get("/health")
    finally:
        main.app.dependency_overrides.clear()
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["qdrant"] == "down"
    assert body["ollama"] == "down"


def test_missing_models_handles_latest_tag() -> None:
    s = Settings(_env_file=None, generator_model="llama3.2", critic_model="qwen3:4b")
    assert main.missing_models(s, {"llama3.2:latest", "qwen3:4b"}) == []
    assert main.missing_models(s, {"gemma2:9b"}) == ["llama3.2", "qwen3:4b"]


@pytest.mark.integration
async def test_health_qdrant_up(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.json()["qdrant"] == "up"
