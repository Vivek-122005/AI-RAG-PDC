"""Two-stage retrieval: broad hybrid recall from the vector store, then precise reranking."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from agentic_rag.core.telemetry import tracer
from agentic_rag.domain.search import SearchHit
from agentic_rag.indexing.store import QdrantStore
from agentic_rag.retrieval.rerank import Reranker

SearchMode = Literal["hybrid", "dense", "sparse"]


@dataclass(frozen=True)
class RetrievalConfig:
    mode: SearchMode = "hybrid"
    candidates: int = 20  # stage 1: how many hits the vector store returns
    top_k: int = 5  # stage 2: how many survive reranking and reach the LLM


@dataclass
class RetrievalResult:
    contexts: list[SearchHit]  # final, reranked
    candidates: list[SearchHit] = field(default_factory=list)  # stage-1 hits (for evaluation)


class EvidenceRetriever(Protocol):
    """What answer pipelines need from retrieval: a query in, ranked evidence out."""

    def retrieve(self, query: str) -> RetrievalResult: ...


class Retriever:
    def __init__(
        self,
        store: QdrantStore,
        reranker: Reranker,
        config: RetrievalConfig | None = None,
        passage_filter: Callable[[str], bool] | None = None,
        tenants: Sequence[str] | None = None,
    ) -> None:
        self.store = store
        self.reranker = reranker
        self.config = config or RetrievalConfig()
        # Documents are untrusted: passages failing the filter (e.g. ones carrying injected
        # instructions) never reach the LLM. Applied at query time, so it also covers
        # chunks indexed before the filter existed.
        self.passage_filter = passage_filter
        self.tenants = tenants  # only search documents these tenants own (None: all)

    def scoped(self, tenants: Sequence[str]) -> "Retriever":
        """The same retriever (shared models and store), restricted to `tenants`' documents."""
        return Retriever(self.store, self.reranker, self.config, self.passage_filter, tenants)

    def retrieve(self, query: str) -> RetrievalResult:
        cfg = self.config
        with tracer().start_as_current_span("retrieval.retrieve") as span:
            candidates = self.store.search(
                query, limit=cfg.candidates, mode=cfg.mode, tenants=self.tenants
            )
            found = len(candidates)
            if self.passage_filter is not None:
                candidates = [h for h in candidates if self.passage_filter(h.text)]
            with tracer().start_as_current_span("retrieval.rerank"):
                contexts = self.reranker.rerank(query, candidates, cfg.top_k)
            span.set_attributes(
                {
                    "rag.retrieval.mode": cfg.mode,
                    "rag.retrieval.candidates": found,
                    "rag.retrieval.filtered_out": found - len(candidates),
                    "rag.retrieval.returned": len(contexts),
                }
            )
            return RetrievalResult(contexts, candidates)
