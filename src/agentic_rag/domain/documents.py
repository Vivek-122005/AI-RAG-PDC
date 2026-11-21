"""Normalized document representation shared by all loaders."""

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Fixed namespace so IDs are stable across runs and machines.
NAMESPACE = uuid.UUID("6f1c2a52-6d1e-4c1b-9a51-2b7a4b0e9d11")


@dataclass(frozen=True)
class Block:
    """A contiguous piece of text (usually a paragraph) with its location."""

    text: str
    page: int | None = None
    section_path: tuple[str, ...] = ()


@dataclass
class Document:
    doc_id: str
    source: str
    title: str
    content_hash: str
    file_type: str
    blocks: list[Block] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    token_count: int
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    flags: tuple[str, ...] = ()  # e.g. ("possible_injection",) set by ingest-time scanning


def doc_id_for(source: str) -> str:
    """Stable document ID derived from the source path (not content), so edits replace."""
    return str(uuid.uuid5(NAMESPACE, source))


def chunk_id_for(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{doc_id}:{chunk_index}"))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 16), b""):
            h.update(part)
    return h.hexdigest()
