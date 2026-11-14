import hashlib
import re
from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient

from agentic_rag.indexing.embedder import SparseVec
from agentic_rag.indexing.store import QdrantStore


class FakeEmbedder:
    """Deterministic, model-free embedder: hashed bag-of-words for both vectors."""

    dense_dim = 64

    def _words(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _bucket(self, word: str, n: int) -> int:
        return int(hashlib.md5(word.encode()).hexdigest(), 16) % n

    def _dense(self, text: str) -> list[float]:
        v = [0.0] * self.dense_dim
        for w in self._words(text):
            v[self._bucket(w, self.dense_dim)] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def _sparse(self, text: str) -> SparseVec:
        counts: dict[int, float] = {}
        for w in self._words(text):
            i = self._bucket(w, 100_000)
            counts[i] = counts.get(i, 0.0) + 1.0
        return SparseVec(list(counts), list(counts.values()))

    def embed_documents(self, texts: Sequence[str]) -> tuple[list[list[float]], list[SparseVec]]:
        return [self._dense(t) for t in texts], [self._sparse(t) for t in texts]

    def embed_query(self, text: str) -> tuple[list[float], SparseVec]:
        return self._dense(text), self._sparse(text)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Everything under tests/integration/ needs running services."""
    for item in items:
        if "integration" in item.path.parts:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def store() -> QdrantStore:
    s = QdrantStore(QdrantClient(":memory:"), "test", FakeEmbedder())
    s.ensure_collection()
    return s
