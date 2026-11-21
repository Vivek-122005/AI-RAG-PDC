"""Application settings, loaded from environment / .env (12-factor)."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    # Vector store. Set QDRANT_PATH to run Qdrant embedded in-process (stored in that folder,
    # no server or Docker needed: the light option for a laptop). Otherwise QDRANT_URL is used.
    # Embedded mode allows one process at a time (it holds a file lock).
    qdrant_path: str | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    # Local resource limits (an 8 GB laptop shares RAM and cores with the LLM)
    onnx_threads: int = 2  # per embedding / reranker model; 0 = all cores
    agent_max_concurrency: int = 1  # parallel retrievals inside one agent run

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

    # Observability: off | console | otlp (see core/telemetry.py)
    tracing: Literal["off", "console", "otlp"] = "off"
    otlp_endpoint: str = "http://localhost:4318"
    service_name: str = "agentic-rag"

    # Guardrails (rule-based: no extra models)
    guardrails_enabled: bool = True
    max_question_chars: int = 1000

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
