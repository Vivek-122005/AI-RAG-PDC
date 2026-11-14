"""`rag` command-line interface."""

import argparse
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_rag import __version__
from agentic_rag.config import Settings, get_settings
from agentic_rag.logging import configure_logging

if TYPE_CHECKING:
    from agentic_rag.indexing.store import QdrantStore


def _store(settings: Settings) -> "QdrantStore":
    # Imported lazily so `rag --help` doesn't load ONNX models.
    from qdrant_client import QdrantClient

    from agentic_rag.indexing.embeddings import FastEmbedEmbedder
    from agentic_rag.indexing.store import QdrantStore

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embedder = FastEmbedEmbedder(settings.dense_model, settings.sparse_model)
    return QdrantStore(client, settings.qdrant_collection, embedder)


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.ingestion.chunking import ChunkingConfig
    from agentic_rag.ingestion.pipeline import ingest_path

    cfg = ChunkingConfig(
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    report = ingest_path(Path(args.path), _store(settings), cfg, force=args.force)
    for source, status in report.results.items():
        print(f"  {status:9} {source}")
    print(report.summary())
    return 1 if report.count("failed") else 0


def cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    store = _store(settings)
    hits = store.search(args.query, limit=args.k, mode=args.mode)
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] score={h.score:.4f}  {h.citation}")
        print(
            textwrap.indent(
                textwrap.fill(h.text[:500] + ("…" if len(h.text) > 500 else ""), 100), "    "
            )
        )
    return 0


def cmd_docs(args: argparse.Namespace, settings: Settings) -> int:
    store = _store(settings)
    store.ensure_collection()
    docs = store.list_documents()
    for d in sorted(docs.values(), key=lambda d: str(d["source"])):
        print(f"  {d['chunk_count']:4} chunks  {d['source']}  —  {d['title']}")
    print(f"{len(docs)} documents, {store.count()} chunks")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rag", description="Agentic RAG CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="ingest a file or directory")
    p.add_argument("path")
    p.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="raw retrieval (no LLM)")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("docs", help="list indexed documents")
    p.set_defaults(func=cmd_docs)

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings)
    sys.exit(args.func(args, settings))
