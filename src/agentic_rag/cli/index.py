"""Corpus management commands: `rag ingest`, `rag docs`."""

import argparse
from pathlib import Path

from agentic_rag.core.config import Settings


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    p = sub.add_parser("ingest", help="ingest a file or directory into the index")
    p.add_argument("path", type=Path)
    p.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    p.set_defaults(handler=ingest)

    p = sub.add_parser("docs", help="list indexed documents")
    p.set_defaults(handler=docs)


def ingest(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.factory import make_store
    from agentic_rag.ingestion.chunking import ChunkingConfig
    from agentic_rag.ingestion.pipeline import ingest_path

    config = ChunkingConfig(
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    report = ingest_path(args.path, make_store(settings), config, force=args.force)
    for source, status in report.results.items():
        print(f"  {status:9} {source}")
    print(report.summary())
    return 1 if report.count("failed") else 0


def docs(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.factory import make_store

    store = make_store(settings)
    store.ensure_collection()
    documents = store.list_documents()
    for d in sorted(documents.values(), key=lambda d: str(d["source"])):
        print(f"  {d['chunk_count']:4} chunks  {d['source']}  -  {d['title']}")
    print(f"{len(documents)} documents, {store.count()} chunks")
    return 0
