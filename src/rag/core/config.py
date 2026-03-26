from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_")

    # OpenAI
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0

    # Vector Store
    vector_store_type: str = "chroma"  # "chroma" or "pgvector"
    chroma_persist_dir: str = "./chroma_langchain_db"
    chroma_collection: str = "documents"
    postgres_dsn: str = "postgresql+psycopg://user:pass@localhost:5432/rag"

    # Chunking
    chunk_size: int = 1500
    chunk_overlap: int = 300

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""
    rate_limit: str = "60/minute"
    max_upload_size_mb: int = 50

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "console"

    # Cache
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000


def get_settings() -> Settings:
    return Settings()
