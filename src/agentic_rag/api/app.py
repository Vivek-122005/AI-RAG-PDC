"""FastAPI application factory.

`create_app()` wires settings, shared services and routes. Tests pass their own `Services`
(in-memory index, fake LLM); production builds them from settings at startup.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from opentelemetry.trace import SpanKind

from agentic_rag import __version__
from agentic_rag.api.routes import documents, health, query
from agentic_rag.api.services import Services
from agentic_rag.api.state import AppState
from agentic_rag.core.config import Settings, get_settings
from agentic_rag.core.logging import configure_logging
from agentic_rag.core.telemetry import configure_from_settings, current_trace_id, tracer

log = structlog.get_logger()
_UI = Path(__file__).with_name("static") / "index.html"


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings)
        configure_from_settings(settings)
        svc = services or Services.from_settings(settings)  # loads models once
        svc.store.ensure_collection()
        app.state.rag = AppState.create(settings, svc)
        log.info("startup", env=settings.app_env, version=__version__)
        yield
        log.info("shutdown")

    app = FastAPI(
        title="Agentic RAG",
        version=__version__,
        description="Evidence-grounded answers with citations from a private document corpus.",
        lifespan=lifespan,
    )
    if origins := [o.strip() for o in settings.cors_origins.split(",") if o.strip()]:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "X-API-Key", "Content-Type"],
        )

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """One server span per request; its trace id is returned as X-Trace-Id."""
        with tracer().start_as_current_span(request.url.path, kind=SpanKind.SERVER) as span:
            span.set_attribute("http.request.method", request.method)
            response = await call_next(request)
            route = request.scope.get("route")
            if path := getattr(route, "path", None):
                span.update_name(f"{request.method} {path}")
                span.set_attribute("http.route", path)
            span.set_attribute("http.response.status_code", response.status_code)
            if trace_id := current_trace_id():
                response.headers["X-Trace-Id"] = trace_id
            return response

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(documents.router)

    @app.get("/", include_in_schema=False)
    def ui() -> HTMLResponse:
        return HTMLResponse(_UI.read_text(encoding="utf-8"))

    return app
