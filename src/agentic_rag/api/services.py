"""Long-lived objects shared by all requests, and per-request pipelines built from them.

Heavy resources (embedding + reranker models, the Qdrant client, LLM clients) are created
once at startup. A request only builds a cheap pipeline object around them, scoped to the
caller's tenant, so one tenant can never retrieve another tenant's documents.
"""

from dataclasses import dataclass, field
from typing import Literal

from agentic_rag import factory
from agentic_rag.core.config import Settings
from agentic_rag.domain.documents import PUBLIC_TENANT
from agentic_rag.guardrails import is_safe_passage
from agentic_rag.indexing.store import QdrantStore
from agentic_rag.llm.base import LLM
from agentic_rag.pipelines.agent import AgentConfig, AgenticRAG
from agentic_rag.pipelines.answer import AnswerPipeline
from agentic_rag.pipelines.baseline import BaselineRAG
from agentic_rag.retrieval.rerank import Reranker
from agentic_rag.retrieval.retriever import RetrievalConfig, Retriever

PipelineKind = Literal["agent", "baseline"]


@dataclass
class Services:
    settings: Settings
    store: QdrantStore
    reranker: Reranker
    llm: LLM
    critic_llm: LLM
    agent_config: AgentConfig = field(default_factory=AgentConfig)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Services":
        llm = factory.make_llm(settings)
        same_model = settings.critic_model == settings.generator_model
        return cls(
            settings=settings,
            store=factory.make_store(settings),
            reranker=factory.make_reranker(settings),
            llm=llm,
            critic_llm=llm if same_model else factory.make_llm(settings, settings.critic_model),
            agent_config=AgentConfig(max_concurrency=settings.agent_max_concurrency),
        )

    @staticmethod
    def readable_tenants(tenant: str) -> list[str]:
        """A tenant reads its own uploads plus the shared public corpus."""
        return sorted({tenant, PUBLIC_TENANT})

    def pipeline(self, tenant: str, kind: PipelineKind = "agent") -> AnswerPipeline:
        s = self.settings
        top_k = self.agent_config.retrieve_depth if kind == "agent" else s.retrieve_top_k
        retriever = Retriever(
            self.store,
            self.reranker,
            RetrievalConfig("hybrid", s.retrieve_candidates, top_k),
            passage_filter=is_safe_passage if s.guardrails_enabled else None,
            tenants=self.readable_tenants(tenant),
        )
        inner: AnswerPipeline = (
            AgenticRAG(retriever, self.llm, self.critic_llm, self.agent_config)
            if kind == "agent"
            else BaselineRAG(retriever, self.llm)
        )
        return factory.make_guarded(s, inner) if s.guardrails_enabled else inner
