"""Two-stage retrieval: broad hybrid recall from the vector store, then precise reranking."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

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
    ) -> None:
        self.store = store
        self.reranker = reranker
        self.config = config or RetrievalConfig()
        # Documents are untrusted: passages failing the filter (e.g. ones carrying injected
        # instructions) never reach the LLM. Applied at query time, so it also covers
        # chunks indexed before the filter existed.
        self.passage_filter = passage_filter

    def retrieve(self, query: str) -> RetrievalResult:
        cfg = self.config
        candidates = self.store.search(query, limit=cfg.candidates, mode=cfg.mode)
        if self.passage_filter is not None:
            candidates = [h for h in candidates if self.passage_filter(h.text)]
        return RetrievalResult(self.reranker.rerank(query, candidates, cfg.top_k), candidates)
