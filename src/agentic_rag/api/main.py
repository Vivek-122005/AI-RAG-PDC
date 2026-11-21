"""ASGI entrypoint: `uvicorn agentic_rag.api.main:app`."""

from agentic_rag.api.app import create_app

app = create_app()
