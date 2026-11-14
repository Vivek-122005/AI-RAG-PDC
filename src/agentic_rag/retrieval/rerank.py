"""Cross-encoder reranking: scores (query, chunk) pairs jointly, far more precise than
bi-encoder similarity, but too slow to run over the whole corpus — so we rerank the
top ~20-30 hybrid candidates."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from agentic_rag.domain.search import SearchHit


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[SearchHit], top_k: int) -> list[SearchHit]: ...


class FastEmbedReranker:
    def __init__(self, model: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = model
        self._encoder = TextCrossEncoder(model)

    def rerank(self, query: str, hits: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        scores = list(self._encoder.rerank(query, [h.text for h in hits]))
        ranked = sorted(zip(hits, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [replace(h, score=float(s)) for h, s in ranked[:top_k]]


class NoopReranker:
    """Keeps the retriever's order (used for ablations)."""

    def rerank(self, query: str, hits: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
        return list(hits[:top_k])
