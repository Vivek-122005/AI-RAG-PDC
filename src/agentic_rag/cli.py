"""Command-line entrypoint. Subcommands (ingest, search, eval) arrive in later phases."""

import argparse

from agentic_rag import __version__


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="Agentic RAG CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args()
    parser.print_help()
