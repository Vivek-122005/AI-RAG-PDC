"""Argument parsing and dispatch. Command modules register their own subcommands.

Heavy dependencies (ONNX models, Qdrant clients) are imported inside command
handlers so `rag --help` stays instant.
"""

import argparse
import sys

from agentic_rag import __version__
from agentic_rag.cli import evaluate, index, query
from agentic_rag.core.config import get_settings
from agentic_rag.core.logging import configure_logging
from agentic_rag.core.telemetry import configure_from_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Agentic RAG with self-correction")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for module in (index, query, evaluate):
        module.register(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings)
    configure_from_settings(settings)
    sys.exit(args.handler(args, settings))
