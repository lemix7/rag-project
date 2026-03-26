
import pytest

from rag.core.config import Settings


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("RAG_OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "2000")
    monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")

    settings = Settings()
    assert settings.openai_api_key == "sk-test-123"
    assert settings.chunk_size == 2000
    assert settings.log_level == "DEBUG"


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("RAG_OPENAI_API_KEY", "sk-test")

    settings = Settings()
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.llm_model == "gpt-4o"
    assert settings.chunk_size == 1500
    assert settings.chunk_overlap == 300
    assert settings.vector_store_type == "chroma"
    assert settings.api_port == 8000
    assert settings.log_format == "json"


def test_settings_missing_api_key(monkeypatch):
    monkeypatch.delenv("RAG_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Clear any .env file influence
    monkeypatch.setattr("pydantic_settings.BaseSettings.model_config", {})

    with pytest.raises(Exception):
        Settings(_env_file=None)
