"""Request and response models of the public HTTP API (v1)."""

from typing import Literal

from pydantic import BaseModel, Field

from agentic_rag.domain.search import SearchHit
from agentic_rag.pipelines.answer import Answer


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    pipeline: Literal["agent", "baseline"] = "agent"


class Citation(BaseModel):
    id: int = Field(description="the [n] used in the answer text")
    cited: bool = Field(description="whether the answer actually cites this passage")
    title: str
    source: str
    section: str | None
    pages: str | None
    snippet: str

    @classmethod
    def from_hit(cls, n: int, hit: SearchHit, cited: bool) -> "Citation":
        pages = None
        if hit.page_start is not None:
            same = hit.page_start == hit.page_end or hit.page_end is None
            pages = str(hit.page_start) if same else f"{hit.page_start}-{hit.page_end}"
        snippet = hit.text if len(hit.text) <= 300 else hit.text[:297] + "..."
        return cls(
            id=n,
            cited=cited,
            title=hit.title,
            source=hit.source,
            section=hit.section_path[-1] if hit.section_path else None,
            pages=pages,
            snippet=snippet,
        )


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    blocked_by: str | None = None
    citations: list[Citation]
    usage: TokenUsage
    timings: dict[str, float]
    trace_id: str | None = None
    cached: bool = False

    @classmethod
    def from_answer(cls, ans: Answer, trace_id: str | None) -> "QueryResponse":
        cited = set(ans.cited_ids)
        return cls(
            answer=ans.text,
            refused=ans.refused,
            blocked_by=ans.blocked_by,
            citations=[Citation.from_hit(n, h, n in cited) for n, h in enumerate(ans.contexts, 1)],
            usage=TokenUsage(
                prompt_tokens=ans.usage.prompt_tokens,
                completion_tokens=ans.usage.completion_tokens,
            ),
            timings={k: round(v, 3) for k, v in ans.timings.items()},
            trace_id=trace_id,
        )


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    source: str
    chunks: int
    owned: bool = Field(description="false for the shared public corpus")
    ingested_at: str | None = None


JobStatus = Literal["queued", "running", "done", "failed"]


class IngestJob(BaseModel):
    job_id: str
    filename: str
    status: JobStatus
    detail: str | None = None
    doc_id: str | None = None
    flagged_chunks: int = 0


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    index: Literal["up", "down"]
    ollama: Literal["up", "down"]
    models_missing: list[str]
