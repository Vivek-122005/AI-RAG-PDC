"""Tracing with OpenTelemetry.

Modules create spans through the OpenTelemetry *API* (`tracer()`), which is a no-op until
`configure_tracing` installs an SDK provider, so tracing costs nothing when it is off.

    TRACING=off       default; no spans recorded
    TRACING=console   print spans to stdout (debugging)
    TRACING=otlp      export to OTLP_ENDPOINT, e.g. the Jaeger UI from
                      `docker compose --profile observability up`

Attribute names follow the OpenTelemetry GenAI conventions where one exists
(`gen_ai.request.model`, `gen_ai.usage.input_tokens`, ...) and use a `rag.` prefix for
this project's own signals (critic decision, loop count, guardrail that fired, ...).
"""

from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from agentic_rag import __version__
from agentic_rag.core.config import Settings

TracingMode = Literal["off", "console", "otlp"]
P = ParamSpec("P")
R = TypeVar("R")

_SCALAR = (str, bool, int, float)


def tracer() -> trace.Tracer:
    return trace.get_tracer("agentic_rag", __version__)


def configure_tracing(
    mode: TracingMode,
    *,
    service_name: str = "agentic-rag",
    environment: str = "dev",
    otlp_endpoint: str = "http://localhost:4318",
) -> None:
    """Install a tracer provider. Call once per process, from an entrypoint."""
    if mode == "off":
        return
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SpanExporter,
    )

    exporter: SpanExporter
    if mode == "console":
        exporter = ConsoleSpanExporter()
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces")
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": __version__,
            "deployment.environment": environment,
        }
    )
    provider = TracerProvider(resource=resource)  # flushes pending spans at exit
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def configure_from_settings(settings: Settings) -> None:
    configure_tracing(
        settings.tracing,
        service_name=settings.service_name,
        environment=settings.app_env,
        otlp_endpoint=settings.otlp_endpoint,
    )


def set_attributes(span: Span, values: Mapping[str, Any], prefix: str = "rag.") -> None:
    """Copy scalars, lists of scalars and (one level of) nested dicts onto a span as
    `prefix + key` attributes; anything else is skipped."""
    for key, value in values.items():
        if isinstance(value, _SCALAR):
            span.set_attribute(prefix + key, value)
        elif isinstance(value, list | tuple) and all(isinstance(v, _SCALAR) for v in value):
            span.set_attribute(prefix + key, [str(v) for v in value])
        elif isinstance(value, Mapping) and prefix.count(".") < 3:
            set_attributes(span, value, f"{prefix}{key}.")


def traced(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Run the function inside a span; exceptions mark the span as an error."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with tracer().start_as_current_span(name) as span:
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        return wrapper

    return decorator


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None
