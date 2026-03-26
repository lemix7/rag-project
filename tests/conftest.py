import pytest

from rag.core.config import Settings


@pytest.fixture
def settings():
    return Settings(
        openai_api_key="test-key-not-real",
        vector_store_type="chroma",
        chroma_persist_dir="/tmp/test_chroma_db",
        chroma_collection="test_collection",
        log_level="DEBUG",
        log_format="console",
    )
