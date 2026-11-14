"""Composition root: the only module that constructs concrete implementations.

Everything else depends on interfaces (Protocols) and receives its collaborators
through constructors, which keeps modules testable and swappable.
"""

from qdrant_client import QdrantClient

from agentic_rag.core.config import Settings
from agentic_rag.indexing.fastembed import FastEmbedEmbedder
from agentic_rag.indexing.store import QdrantStore
from agentic_rag.llm.ollama import OllamaLLM
from agentic_rag.pipelines.baseline import BaselineRAG
from agentic_rag.retrieval.rerank import FastEmbedReranker, NoopReranker, Reranker
from agentic_rag.retrieval.retriever import RetrievalConfig, Retriever, SearchMode


def make_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def make_store(settings: Settings) -> QdrantStore:
    embedder = FastEmbedEmbedder(settings.dense_model, settings.sparse_model)
    return QdrantStore(make_qdrant_client(settings), settings.qdrant_collection, embedder)


def make_reranker(settings: Settings, enabled: bool = True, model: str | None = None) -> Reranker:
    if not enabled:
        return NoopReranker()
    return FastEmbedReranker(model or settings.reranker_model)


def make_retriever(
    settings: Settings,
    store: QdrantStore | None = None,
    rerank: bool = True,
    mode: SearchMode = "hybrid",
) -> Retriever:
    config = RetrievalConfig(mode, settings.retrieve_candidates, settings.retrieve_top_k)
    return Retriever(store or make_store(settings), make_reranker(settings, rerank), config)


def make_llm(settings: Settings, model: str | None = None) -> OllamaLLM:
    return OllamaLLM(
        model or settings.generator_model,
        base_url=settings.ollama_base_url,
        num_ctx=settings.llm_num_ctx,
        timeout_s=settings.llm_timeout_s,
    )


def make_baseline(
    settings: Settings, rerank: bool = True, mode: SearchMode = "hybrid"
) -> BaselineRAG:
    return BaselineRAG(make_retriever(settings, rerank=rerank, mode=mode), make_llm(settings))
