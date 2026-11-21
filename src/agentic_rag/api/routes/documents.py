"""Document management: upload (background ingest), list, job status, delete."""

import re
from pathlib import Path
from typing import Any

import anyio
import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, UploadFile, status

from agentic_rag.api.schemas import DocumentInfo, IngestJob
from agentic_rag.api.state import AppState, LimitedTenantDep, StateDep, TenantDep
from agentic_rag.domain.documents import PUBLIC_TENANT, doc_id_for
from agentic_rag.factory import make_chunking_config
from agentic_rag.ingestion.loaders import SUPPORTED_SUFFIXES
from agentic_rag.ingestion.pipeline import ingest_path

router = APIRouter(prefix="/v1/documents", tags=["documents"])
log = structlog.get_logger()

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_CHUNK = 1 << 20


def safe_filename(name: str | None) -> str:
    """A plain file name: no directories, no traversal, only portable characters."""
    base = _UNSAFE.sub("_", Path(name or "").name).strip("._")
    if not base:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "file needs a name")
    return base[:120]


def _upload_path(state: AppState, tenant: str, filename: str) -> Path:
    return Path(state.settings.upload_dir) / tenant / filename


def _owner(payload: dict[str, Any]) -> str:
    return str(payload.get("tenant") or PUBLIC_TENANT)  # pre-tenancy chunks are public


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload(
    file: UploadFile, background: BackgroundTasks, state: StateDep, tenant: LimitedTenantDep
) -> IngestJob:
    """Store a PDF / Markdown / HTML / text file and index it in the background.
    Poll `GET /v1/documents/jobs/{job_id}` for the result."""
    name = safe_filename(file.filename)
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"supported: {allowed}")

    dest = _upload_path(state, tenant, name)
    limit = state.settings.max_upload_mb * _CHUNK
    data = bytearray()
    while chunk := await file.read(_CHUNK):
        data += chunk
        if len(data) > limit:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"files are limited to {state.settings.max_upload_mb} MB",
            )
    await anyio.Path(dest.parent).mkdir(parents=True, exist_ok=True)
    await anyio.Path(dest).write_bytes(bytes(data))

    job = state.jobs.create(tenant, name)
    background.add_task(run_ingest, state, job.job_id, dest, tenant)
    return job


async def run_ingest(state: AppState, job_id: str, path: Path, tenant: str) -> None:
    state.jobs.update(job_id, "running")
    store, config = state.services.store, make_chunking_config(state.settings)
    try:
        report = await anyio.to_thread.run_sync(
            lambda: ingest_path(path, store, config, tenant=tenant), limiter=state.work
        )
    except Exception as exc:  # report every failure on the job instead of losing it
        log.error("ingest_job_failed", job_id=job_id, error=str(exc))
        state.jobs.update(job_id, "failed", detail=str(exc))
        return
    outcome = report.results.get(path.name, "failed")
    state.cache.clear()  # any answer may change once the corpus does
    if outcome in ("failed", "empty"):
        detail = "no text could be extracted" if outcome == "empty" else "could not be parsed"
        state.jobs.update(job_id, "failed", detail=detail)
    else:
        state.jobs.update(
            job_id,
            "done",
            detail=outcome,
            doc_id=doc_id_for(path.name, tenant),
            flagged_chunks=len(report.flagged),
        )


@router.get("/jobs/{job_id}")
def job_status(job_id: str, state: StateDep, tenant: TenantDep) -> IngestJob:
    if (job := state.jobs.get(job_id, tenant)) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such job")
    return job


@router.get("")
async def list_documents(state: StateDep, tenant: TenantDep) -> list[DocumentInfo]:
    """Documents the caller can read: its own uploads and the shared public corpus."""
    readable = state.services.readable_tenants(tenant)
    docs = await anyio.to_thread.run_sync(lambda: state.services.store.list_documents(readable))
    return [
        DocumentInfo(
            doc_id=doc_id,
            title=str(p.get("title", "")),
            source=str(p.get("source", "")),
            chunks=int(p.get("chunk_count", 0)),
            owned=_owner(p) == tenant,
            ingested_at=p.get("ingested_at"),
        )
        for doc_id, p in sorted(docs.items(), key=lambda kv: str(kv[1].get("source", "")))
    ]


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str, state: StateDep, tenant: TenantDep) -> Response:
    store = state.services.store
    payload = await anyio.to_thread.run_sync(lambda: store.get_document(doc_id))
    owner = _owner(payload) if payload else None
    if owner is None or owner not in state.services.readable_tenants(tenant):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such document")  # don't leak others'
    if owner != tenant:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "shared documents are managed by operators")
    await anyio.to_thread.run_sync(lambda: store.delete_document(doc_id))
    upload = _upload_path(state, tenant, str(payload.get("source", ""))) if payload else None
    if upload is not None and await anyio.Path(upload).is_file():
        await anyio.Path(upload).unlink()
    state.cache.clear()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
