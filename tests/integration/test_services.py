"""API against a real Qdrant server (`make up`): health, and tenant isolation on the
server's filter engine. The LLM and embedder are fakes, so no Ollama is needed."""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from conftest import FakeEmbedder
from fakes import FakeLLM
from httpx import ASGITransport, AsyncClient
from qdrant_client import QdrantClient

from agentic_rag.api.app import create_app
from agentic_rag.api.services import Services
from agentic_rag.core.config import Settings
from agentic_rag.indexing.store import QdrantStore
from agentic_rag.ingestion.chunking import ChunkingConfig
from agentic_rag.ingestion.pipeline import ingest_path
from agentic_rag.retrieval import NoopReranker

ALICE, BOB = {"X-API-Key": "alice-key"}, {"X-API-Key": "bob-key"}


@pytest.fixture
async def http(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, FakeLLM]]:
    settings = Settings(
        _env_file=None,
        api_keys="alice-key:alice,bob-key:bob",
        upload_dir=str(tmp_path / "uploads"),
        ollama_base_url="http://127.0.0.1:1",
    )
    client = QdrantClient(url=settings.qdrant_url)
    collection = f"it-{uuid.uuid4().hex[:8]}"
    store = QdrantStore(client, collection, FakeEmbedder())
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "crag.md").write_text("# CRAG\n\nCRAG triggers Correct, Incorrect, Ambiguous.\n")
    ingest_path(corpus, store, ChunkingConfig(min_tokens=1))

    llm = FakeLLM(lambda prompt: "From the documents [1].")
    app = create_app(settings, Services(settings, store, NoopReranker(), llm, llm))
    try:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                yield c, llm
    finally:
        client.delete_collection(collection)


async def test_health_reports_the_index_up(http: tuple[AsyncClient, FakeLLM]) -> None:
    client, _ = http
    assert (await client.get("/health")).json()["index"] == "up"


async def test_tenant_filter_on_a_qdrant_server(http: tuple[AsyncClient, FakeLLM]) -> None:
    client, llm = http
    files = {"file": ("notes.md", b"# Notes\n\nThe codename is Bluebird.\n", "text/markdown")}
    assert (await client.post("/v1/documents", headers=ALICE, files=files)).status_code == 202

    def seen() -> str:
        return "\n".join(m.content for call in llm.calls for m in call)

    body = {"question": "What is the codename?", "pipeline": "baseline"}
    await client.post("/v1/query", json=body, headers=BOB)
    assert "Bluebird" not in seen() and "CRAG" in seen()  # public visible, alice's not
    await client.post("/v1/query", json=body, headers=ALICE)
    assert "Bluebird" in seen()
