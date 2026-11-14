import pytest

from agentic_rag.core.config import Settings


def test_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.qdrant_collection == "documents"
    assert s.app_env == "dev"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("APP_ENV", "prod")
    s = Settings(_env_file=None)
    assert s.qdrant_url == "http://qdrant:6333"
    assert s.app_env == "prod"
