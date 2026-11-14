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

    # Embeddings / chunking
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 450
    chunk_overlap_tokens: int = 50

    # Retrieval
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-12-v2"
    retrieve_candidates: int = 20  # hybrid candidates passed to the reranker
    retrieve_top_k: int = 5  # chunks given to the LLM

    # LLM (local Ollama, running natively on the host)
    ollama_base_url: str = "http://localhost:11434"
    generator_model: str = "qwen3:4b-instruct"
    critic_model: str = "qwen3:4b-instruct"
    judge_model: str = "gemma3:4b"  # offline evaluation only; different family from generator
    llm_num_ctx: int = 8192
    llm_timeout_s: float = 120.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
