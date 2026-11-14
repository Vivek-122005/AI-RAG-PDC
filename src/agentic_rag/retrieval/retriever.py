"""Two-stage retrieval: broad hybrid recall from the vector store, then precise reranking."""

from dataclasses import dataclass, field
from typing import Literal

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


class Retriever:
    def __init__(
        self, store: QdrantStore, reranker: Reranker, config: RetrievalConfig | None = None
    ) -> None:
        self.store = store
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    def retrieve(self, query: str) -> RetrievalResult:
        cfg = self.config
        candidates = self.store.search(query, limit=cfg.candidates, mode=cfg.mode)
        return RetrievalResult(self.reranker.rerank(query, candidates, cfg.top_k), candidates)
