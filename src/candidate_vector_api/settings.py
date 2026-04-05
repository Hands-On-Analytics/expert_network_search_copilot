from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    postgres_host: str = Field(..., alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(..., alias="POSTGRES_DB")
    postgres_user: str = Field(..., alias="POSTGRES_USER")
    postgres_password: str = Field(..., alias="POSTGRES_PASSWORD")

    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field("https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    embedding_model: str = Field("openai/text-embedding-3-small", alias="EMBEDDING_MODEL")
    openrouter_site_url: str | None = Field(default=None, alias="OPENROUTER_SITE_URL")
    openrouter_app_name: str | None = Field(default="candidate-vector-api", alias="OPENROUTER_APP_NAME")

    chroma_path: Path = Field(default=Path(".chroma"), alias="CHROMA_PATH")
    chroma_collection_name: str = Field("candidate_profiles", alias="CHROMA_COLLECTION_NAME")
    embedding_batch_size: int = Field(64, alias="EMBEDDING_BATCH_SIZE", ge=1, le=256)
    upsert_batch_size: int = Field(128, alias="UPSERT_BATCH_SIZE", ge=1, le=512)
    max_chunk_chars: int = Field(1800, alias="MAX_CHUNK_CHARS", ge=200, le=8000)
    search_candidate_pool_size: int = Field(250, alias="SEARCH_CANDIDATE_POOL_SIZE", ge=20, le=2000)
    search_max_query_variants: int = Field(4, alias="SEARCH_MAX_QUERY_VARIANTS", ge=1, le=10)
    search_highlight_char_limit: int = Field(260, alias="SEARCH_HIGHLIGHT_CHAR_LIMIT", ge=80, le=1200)

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def postgres_url(self) -> str:
        encoded_password = quote_plus(self.postgres_password)
        return (
            f"postgresql://{self.postgres_user}:{encoded_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def resolved_chroma_path(self) -> Path:
        if self.chroma_path.is_absolute():
            return self.chroma_path
        return PROJECT_ROOT / self.chroma_path


@lru_cache
def get_settings() -> Settings:
    return Settings()
