"""Embedding interface used by the vector store. Implementations live in sibling modules."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


class Embedder(Protocol):
    dense_dim: int

    def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], list[SparseVec]]: ...

    def embed_query(self, text: str) -> tuple[list[float], SparseVec]: ...
