"""Application settings, loaded from environment / .env (12-factor)."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    # LLM (local Ollama, running natively on the host)
    ollama_base_url: str = "http://localhost:11434"
    generator_model: str = "qwen3:4b"
    critic_model: str = "qwen3:4b"
    judge_model: str = "gemma2:9b"  # offline evaluation only
    llm_num_ctx: int = 8192
    llm_timeout_s: float = 120.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
