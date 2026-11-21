"""Composition root: the only module that constructs concrete implementations.

Everything else depends on interfaces (Protocols) and receives its collaborators
through constructors, which keeps modules testable and swappable.
"""

from qdrant_client import QdrantClient

from agentic_rag.core.config import Settings
from agentic_rag.guardrails import InputGuard, OutputGuard, is_safe_passage
from agentic_rag.indexing.fastembed import FastEmbedEmbedder
from agentic_rag.indexing.store import QdrantStore
from agentic_rag.llm.ollama import OllamaLLM
from agentic_rag.pipelines.agent import AgentConfig, AgenticRAG
from agentic_rag.pipelines.agent.prompts import CRITIC_SYSTEM, DRAFT_SYSTEM, PLANNER_SYSTEM
from agentic_rag.pipelines.answer import AnswerPipeline
from agentic_rag.pipelines.baseline import BaselineRAG
from agentic_rag.pipelines.guarded import GuardedPipeline
from agentic_rag.pipelines.prompts import ANSWER_SYSTEM
from agentic_rag.retrieval.rerank import FastEmbedReranker, NoopReranker, Reranker
from agentic_rag.retrieval.retriever import RetrievalConfig, Retriever, SearchMode


def make_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_path:  # embedded: no server process
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def _threads(settings: Settings) -> int | None:
    return settings.onnx_threads or None  # 0 → let ONNX Runtime use every core


def make_store(settings: Settings) -> QdrantStore:
    embedder = FastEmbedEmbedder(
        settings.dense_model, settings.sparse_model, threads=_threads(settings)
    )
    return QdrantStore(make_qdrant_client(settings), settings.qdrant_collection, embedder)


def make_reranker(settings: Settings, enabled: bool = True, model: str | None = None) -> Reranker:
    if not enabled:
        return NoopReranker()
    return FastEmbedReranker(model or settings.reranker_model, threads=_threads(settings))


def make_retriever(
    settings: Settings,
    store: QdrantStore | None = None,
    rerank: bool = True,
    mode: SearchMode = "hybrid",
    top_k: int | None = None,
    guardrails: bool | None = None,
) -> Retriever:
    config = RetrievalConfig(mode, settings.retrieve_candidates, top_k or settings.retrieve_top_k)
    guarded = settings.guardrails_enabled if guardrails is None else guardrails
    return Retriever(
        store or make_store(settings),
        make_reranker(settings, rerank),
        config,
        passage_filter=is_safe_passage if guarded else None,  # documents are untrusted
    )


def make_llm(settings: Settings, model: str | None = None) -> OllamaLLM:
    return OllamaLLM(
        model or settings.generator_model,
        base_url=settings.ollama_base_url,
        num_ctx=settings.llm_num_ctx,
        timeout_s=settings.llm_timeout_s,
    )


def make_guarded(settings: Settings, pipeline: AnswerPipeline) -> GuardedPipeline:
    return GuardedPipeline(
        pipeline,
        InputGuard(max_chars=settings.max_question_chars),
        OutputGuard(system_prompts=(ANSWER_SYSTEM, DRAFT_SYSTEM, PLANNER_SYSTEM, CRITIC_SYSTEM)),
    )


def make_baseline(
    settings: Settings,
    rerank: bool = True,
    mode: SearchMode = "hybrid",
    guardrails: bool | None = None,
) -> AnswerPipeline:
    guarded = settings.guardrails_enabled if guardrails is None else guardrails
    retriever = make_retriever(settings, rerank=rerank, mode=mode, guardrails=guarded)
    rag = BaselineRAG(retriever, make_llm(settings))
    return make_guarded(settings, rag) if guarded else rag


def make_agent(
    settings: Settings,
    config: AgentConfig | None = None,
    rerank: bool = True,
    mode: SearchMode = "hybrid",
    guardrails: bool | None = None,
) -> AnswerPipeline:
    generator = make_llm(settings)
    # Reuse one client when the critic is the same model, so Ollama keeps a single copy loaded.
    critic = (
        generator
        if settings.critic_model == settings.generator_model
        else make_llm(settings, settings.critic_model)
    )
    config = config or AgentConfig(max_concurrency=settings.agent_max_concurrency)
    guarded = settings.guardrails_enabled if guardrails is None else guardrails
    retriever = make_retriever(
        settings, rerank=rerank, mode=mode, top_k=config.retrieve_depth, guardrails=guarded
    )
    agent = AgenticRAG(retriever, generator, critic, config)
    return make_guarded(settings, agent) if guarded else agent
