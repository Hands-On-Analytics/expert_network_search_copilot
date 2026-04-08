from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "expert-network-copilot"
    APP_ENV: str = "dev"
    LOG_LEVEL: str = "INFO"

    POSTGRES_DSN: str

    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "candidate_profiles"

    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
    OPENROUTER_CHAT_MODEL: str = "openai/gpt-4o-mini"

    RAG_TOP_K: int = 5

    INGEST_BATCH_SIZE: int = 200
    # EMBED_BATCH_SIZE: int = 64
    # MAX_CONCURRENCY: int = 8


settings = Settings()