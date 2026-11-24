"""Settings, read from environment variables and the `.env` file.

Every setting can be overridden with an environment variable prefixed `NTMB_`,
e.g. `NTMB_LLM_MODEL=openai/gpt-4o-mini`. Settings are added here as each task needs them.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NTMB_", env_file=".env", extra="ignore")

    # Vector DB (Weaviate, started with `docker compose up -d`)
    weaviate_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051

    # Chunking (defaults derived from the starter corpus, see chunking.chunk_pages)
    chunk_max_tokens: int = 320
    chunk_overlap_tokens: int = 75
    chunk_min_tokens: int = 65

    # Embeddings. bge-small: 512-token input (no chunk truncated), 384 dims, fast on CPU/MPS.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # BGE models are trained to embed short queries with this instruction in front;
    # documents are embedded as they are.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Retrieval. Provisional defaults: on hand-checked questions, hybrid was never worse than
    # the better of semantic and BM25. PER-15 tunes them against the eval set.
    retrieval_mode: Literal["semantic", "bm25", "hybrid"] = "hybrid"
    retrieval_k: int = 5
    retrieval_alpha: float = 0.5  # 1.0 = pure semantic, 0.0 = pure BM25
    retrieval_fusion: Literal["relative_score", "ranked"] = "relative_score"

    # LLM, in LiteLLM's "provider/model" format
    # `ollama_chat/` is LiteLLM's chat endpoint for Ollama (supports system prompts properly).
    llm_model: str = "ollama_chat/qwen3:14b"
    llm_temperature: float = 0.1  # low: report what the sources say, no creative variation
    # Local thinking models reason before answering: ~10x slower (14s vs 1.4s measured).
    llm_think: bool = False

    # Only needed when llm_model points to OpenAI. SecretStr hides it in logs and reprs.
    openai_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
