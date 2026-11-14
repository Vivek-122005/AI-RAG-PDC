"""Domain types shared across layers (documents, chunks, search hits). No I/O, no deps."""

from agentic_rag.domain.documents import Block, Chunk, Document
from agentic_rag.domain.search import SearchHit

__all__ = ["Block", "Chunk", "Document", "SearchHit"]
