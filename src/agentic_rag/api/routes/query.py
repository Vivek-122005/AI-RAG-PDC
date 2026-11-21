"""POST /v1/query (JSON) and POST /v1/query/stream (Server-Sent Events)."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
import structlog
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from agentic_rag.api.schemas import QueryRequest, QueryResponse
from agentic_rag.api.state import AppState, LimitedTenantDep, StateDep
from agentic_rag.core.telemetry import current_trace_id
from agentic_rag.llm.base import LLMError
from agentic_rag.pipelines.answer import StepCallback

router = APIRouter(prefix="/v1", tags=["query"])
log = structlog.get_logger()

# Step fields worth showing a client while it waits; the rest (chunk ids, scores) is noise.
_STATUS_FIELDS = ("node", "iteration", "decision", "refused", "strategy", "queries", "blocked")


async def answer_query(
    state: AppState, tenant: str, req: QueryRequest, on_step: StepCallback | None = None
) -> QueryResponse:
    if (hit := state.cache.get(tenant, req.pipeline, req.question)) is not None:
        return QueryResponse.model_validate({**hit, "cached": True})

    pipeline = state.services.pipeline(tenant, req.pipeline)
    trace_id = current_trace_id()  # the request span; pipeline spans become its children
    try:
        # Pipelines are synchronous and heavy: run them on a worker thread, and through the
        # shared limiter so a laptop handles one at a time while the event loop stays free.
        ans = await anyio.to_thread.run_sync(
            lambda: pipeline.answer(req.question, on_step), limiter=state.work
        )
    except LLMError as exc:
        log.error("llm_unavailable", error=str(exc))
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "language model unavailable"
        ) from exc

    resp = QueryResponse.from_answer(ans, trace_id)
    if ans.blocked_by is None or ans.blocked_by.startswith("output:"):
        state.cache.put(tenant, req.pipeline, req.question, resp.model_dump(exclude={"cached"}))
    return resp


@router.post("/query")
async def query(req: QueryRequest, state: StateDep, tenant: LimitedTenantDep) -> QueryResponse:
    """Answer a question from the documents the caller can read, with citations."""
    return await answer_query(state, tenant, req)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/query/stream", response_class=StreamingResponse)
async def query_stream(
    req: QueryRequest, state: StateDep, tenant: LimitedTenantDep
) -> StreamingResponse:
    """Same as /v1/query, as Server-Sent Events:

    - `status`    one per pipeline step as it finishes (plan, retrieve, draft, critique, ...)
    - `citations` the passages the answer may cite
    - `final`     the full QueryResponse
    - `error`     instead of `final` if the model is unavailable

    Draft tokens are not streamed: the critic may still reject a draft, and showing text
    that is then withdrawn would defeat the point of self-correction.
    """
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_step(step: dict[str, Any]) -> None:  # called on the worker thread
        public = {k: step[k] for k in _STATUS_FIELDS if step.get(k) not in (None, [], "")}
        loop.call_soon_threadsafe(queue.put_nowait, ("status", public))

    async def produce() -> None:
        try:
            resp = await answer_query(state, tenant, req, on_step)
            citations = [c.model_dump() for c in resp.citations]
            await queue.put(("citations", {"citations": citations}))
            await queue.put(("final", resp.model_dump()))
        except HTTPException as exc:
            await queue.put(("error", {"status": exc.status_code, "detail": exc.detail}))
        finally:
            await queue.put(None)

    async def events() -> AsyncIterator[str]:
        task = asyncio.create_task(produce())
        try:
            while (item := await queue.get()) is not None:
                yield _sse(*item)
        finally:
            task.cancel()  # client went away; a running pipeline still finishes its thread

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
