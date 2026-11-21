"""API-key authentication (key → tenant) and per-tenant rate limiting.

In-memory on purpose: with embedded Qdrant the API is a single process, so a Redis
instance would add a container without adding correctness. Swap `RateLimiter` for a
Redis-backed one when running several API replicas against a Qdrant server.
"""

import hmac
import threading
import time
from dataclasses import dataclass, field

from agentic_rag.core.config import Settings

DEV_TENANT = "local"


class AuthConfigError(RuntimeError):
    pass


def parse_api_keys(raw: str) -> dict[str, str]:
    """`"key1:alice,key2:bob"` → {"key1": "alice", "key2": "bob"}."""
    keys: dict[str, str] = {}
    for entry in filter(None, (e.strip() for e in raw.split(","))):
        key, sep, tenant = entry.partition(":")
        if not sep or not key.strip() or not tenant.strip():
            raise AuthConfigError(f"API_KEYS entries must look like key:tenant, got {entry!r}")
        if tenant.strip() == "public":
            raise AuthConfigError("tenant name 'public' is reserved for the shared corpus")
        keys[key.strip()] = tenant.strip()
    return keys


@dataclass
class Authenticator:
    keys: dict[str, str]
    open_access: bool  # dev convenience: no keys configured → every caller is DEV_TENANT

    @classmethod
    def from_settings(cls, settings: Settings) -> "Authenticator":
        keys = parse_api_keys(settings.api_keys.get_secret_value())
        if not keys and settings.app_env == "prod":
            raise AuthConfigError("APP_ENV=prod requires API_KEYS")
        return cls(keys, open_access=not keys)

    def tenant_for(self, presented: str | None) -> str | None:
        """The tenant a key belongs to, or None. Constant-time comparison per key."""
        if self.open_access:
            return DEV_TENANT
        if not presented:
            return None
        for key, tenant in self.keys.items():
            if hmac.compare_digest(key.encode(), presented.encode()):
                return tenant
        return None


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass
class RateLimiter:
    """Token bucket per tenant: `per_minute` requests, refilled continuously."""

    per_minute: int
    _buckets: dict[str, _Bucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire(self, tenant: str, now: float | None = None) -> float:
        """Take one token. Returns 0 if allowed, else seconds until a token is available."""
        now = time.monotonic() if now is None else now
        rate = self.per_minute / 60.0
        with self._lock:
            b = self._buckets.setdefault(tenant, _Bucket(float(self.per_minute), now))
            b.tokens = min(float(self.per_minute), b.tokens + (now - b.updated) * rate)
            b.updated = now
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return 0.0
            return (1.0 - b.tokens) / rate
