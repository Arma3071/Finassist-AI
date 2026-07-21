"""Application configuration.

Centralizes all environment-driven settings for FinAssist AI: LLM provider
selection, embedding model, chunking parameters, vector store location,
and third-party financial API keys.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider ---------------------------------------------------
    llm_provider: Literal["openai", "anthropic"] = Field(
        default="anthropic", description="Which LLM backend to use."
    )
    openai_api_key: str = Field(default="", description="OpenAI API key.")
    openai_model: str = Field(default="gpt-4o-mini")
    anthropic_api_key: str = Field(default="", description="Anthropic API key.")
    anthropic_model: str = Field(default="claude-sonnet-4-20250514")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=1.0)

    # --- Embeddings -------------------------------------------------------
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5")

    # --- Chunking -----------------------------------------------------
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    chunking_strategy: Literal["recursive", "semantic"] = Field(default="recursive")

    # --- Vector store -----------------------------------------------------
    chroma_persist_dir: str = Field(default="./data/chroma")
    chroma_collection_name: str = Field(default="finassist_documents")

    # --- Retrieval ----------------------------------------------------
    retrieval_top_k: int = Field(default=5, gt=0)
    retrieval_search_type: Literal["similarity", "mmr"] = Field(default="mmr")
    mmr_fetch_k: int = Field(default=20, gt=0)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- Financial / external APIs ---------------------------------------
    alpha_vantage_api_key: str = Field(default="")
    exchangerate_api_key: str = Field(default="")
    newsapi_api_key: str = Field(default="")

    # --- CORS -----------------------------------------------------------
    cors_origins: str = Field(default="http://localhost:8501,http://127.0.0.1:8501", description="Comma-separated allowed CORS origins.")

    # --- Misc ---------------------------------------------------------
    log_level: str = Field(default="INFO")
    max_conversation_turns: int = Field(default=12, gt=0)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
