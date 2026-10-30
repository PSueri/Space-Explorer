from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    name: str = Field(default="Space Explorer", min_length=1)
    debug: bool = False

    llm_provider: Literal["ollama"] = "ollama"
    llm_model: str = Field(default="qwen3:0.6b", min_length=1)
    llm_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    llm_context_size: int = Field(default=2048, ge=512, le=8192)
    llm_max_output_tokens: int = Field(default=128, ge=1, le=1024)

    embedding_model: str = Field(default="all-minilm:22m", min_length=1)
    embedding_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    embedding_timeout_seconds: float = Field(default=60.0, gt=0, le=300)

    rag_diagnostics_enabled: bool = False
