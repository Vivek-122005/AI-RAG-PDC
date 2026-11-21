"""Idempotent ingest: load → chunk → embed → upsert, skipping unchanged files."""

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import structlog

from agentic_rag.domain.documents import Chunk, doc_id_for, file_hash
from agentic_rag.guardrails.injection import is_safe_passage
from agentic_rag.indexing.store import QdrantStore
from agentic_rag.ingestion.chunking import ChunkingConfig, chunk_document, embedding_text
from agentic_rag.ingestion.loaders import SUPPORTED_SUFFIXES, load_document

log = structlog.get_logger()

Status = Literal["added", "updated", "unchanged", "failed", "empty"]


@dataclass
class IngestReport:
    results: dict[str, Status] = field(default_factory=dict)
    chunks_written: int = 0
    flagged: list[tuple[str, int]] = field(default_factory=list)  # (source, chunk_index)

    def count(self, status: Status) -> int:
        return sum(1 for s in self.results.values() if s == status)

    def summary(self) -> str:
        parts = [
            f"{s}={self.count(s)}" for s in ("added", "updated", "unchanged", "empty", "failed")
        ]
        return " ".join(parts) + (
            f" chunks_written={self.chunks_written} flagged_chunks={len(self.flagged)}"
        )


def _scan(chunk: Chunk) -> Chunk:
    """Documents are untrusted: flag chunks that look like instructions aimed at the model.
    Retrieval drops such passages anyway; the flag makes them visible at ingest time."""
    return (
        replace(chunk, flags=("possible_injection",)) if not is_safe_passage(chunk.text) else chunk
    )


def iter_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith("."):
            yield p


def source_for(path: Path, root: Path) -> str:
    """Stable source ID: path relative to the ingest root (portable across machines)."""
    base = root.parent if root.is_file() else root
    return path.resolve().relative_to(base.resolve()).as_posix()


def ingest_path(
    root: Path, store: QdrantStore, cfg: ChunkingConfig | None = None, force: bool = False
) -> IngestReport:
    store.ensure_collection()
    report = IngestReport()
    for path in iter_files(root):
        source = source_for(path, root)
        try:
            existing = store.get_content_hash(doc_id_for(source))
            if not force and existing == file_hash(path):
                report.results[source] = "unchanged"
                continue
            doc = load_document(path, source=source)
            chunks = [_scan(c) for c in chunk_document(doc, cfg)]
            if not chunks:
                report.results[source] = "empty"
                log.warning("no_text_extracted", source=source)
                continue
            store.upsert_document(doc, chunks, [embedding_text(doc.title, c) for c in chunks])
            report.results[source] = "updated" if existing else "added"
            report.chunks_written += len(chunks)
            report.flagged += [(source, c.chunk_index) for c in chunks if c.flags]
            log.info("ingested", source=source, chunks=len(chunks), title=doc.title)
        except Exception as exc:  # one bad file must not abort the batch
            report.results[source] = "failed"
            log.error("ingest_failed", source=source, error=str(exc))
    return report
