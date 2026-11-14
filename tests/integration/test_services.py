"""Requires `make up` (Qdrant) and a running Ollama."""

from httpx import ASGITransport, AsyncClient

from agentic_rag.api.main import app


async def test_health_reports_qdrant_up() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        body = (await client.get("/health")).json()
    assert body["qdrant"] == "up"
