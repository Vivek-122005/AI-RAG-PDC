"""Process-wide API state and the FastAPI dependencies that hand it to routes."""

import threading
import uuid
from dataclasses import dataclass, field
from math import ceil
from typing import Annotated

import anyio
from fastapi import Depends, Header, HTTPException, Request, status

from agentic_rag.api.cache import AnswerCache
from agentic_rag.api.schemas import IngestJob, JobStatus
from agentic_rag.api.security import Authenticator, RateLimiter
from agentic_rag.api.services import Services
from agentic_rag.core.config import Settings


@dataclass
class JobRegistry:
    """Ingest jobs of this process (in memory: jobs do not survive a restart)."""

    _jobs: dict[str, tuple[str, IngestJob]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def create(self, tenant: str, filename: str) -> IngestJob:
        job = IngestJob(job_id=uuid.uuid4().hex, filename=filename, status="queued")
        with self._lock:
            self._jobs[job.job_id] = (tenant, job)
        return job

    def update(self, job_id: str, status: JobStatus, **fields: object) -> None:
        with self._lock:
            tenant, job = self._jobs[job_id]
            self._jobs[job_id] = (tenant, job.model_copy(update={"status": status, **fields}))

    def get(self, job_id: str, tenant: str) -> IngestJob | None:
        with self._lock:
            owner, job = self._jobs.get(job_id, ("", None))
        return job if owner == tenant else None


@dataclass
class AppState:
    settings: Settings
    services: Services
    auth: Authenticator
    limiter: RateLimiter
    cache: AnswerCache
    jobs: JobRegistry
    # One limiter for every heavy job (queries and ingests): on a laptop, one at a time.
    work: anyio.CapacityLimiter

    @classmethod
    def create(cls, settings: Settings, services: Services) -> "AppState":
        return cls(
            settings=settings,
            services=services,
            auth=Authenticator.from_settings(settings),
            limiter=RateLimiter(settings.api_rate_limit_per_minute),
            cache=AnswerCache(settings.answer_cache_size, settings.answer_cache_ttl_s),
            jobs=JobRegistry(),
            work=anyio.CapacityLimiter(settings.api_max_concurrent_jobs),
        )


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.rag
    return state


StateDep = Annotated[AppState, Depends(get_state)]


def _presented_key(authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def get_tenant(
    state: StateDep,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> str:
    tenant = state.auth.tenant_for(_presented_key(authorization, x_api_key))
    if tenant is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return tenant


TenantDep = Annotated[str, Depends(get_tenant)]


def rate_limited_tenant(state: StateDep, tenant: TenantDep) -> str:
    """Authenticated tenant that still has request budget (for costly endpoints)."""
    if (wait := state.limiter.acquire(tenant)) > 0:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate limit exceeded",
            headers={"Retry-After": str(ceil(wait))},
        )
    return tenant


LimitedTenantDep = Annotated[str, Depends(rate_limited_tenant)]
