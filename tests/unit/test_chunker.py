from langchain_core.documents import Document

from rag.core.config import Settings
from rag.ingestion.chunker import chunk_documents


def _make_settings(**overrides) -> Settings:
    defaults = {"openai_api_key": "test-key", "chunk_size": 100, "chunk_overlap": 20}
    defaults.update(overrides)
    return Settings(**defaults)


def test_chunk_documents_splits_long_text():
    text = "word " * 200  # ~1000 chars
    docs = [Document(page_content=text, metadata={"source": "test.pdf", "page": 0})]
    settings = _make_settings(chunk_size=100, chunk_overlap=20)

    chunks = chunk_documents(docs, settings)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 100


def test_chunk_documents_preserves_metadata():
    meta = {"source": "test.pdf", "page": 3}
    docs = [Document(page_content="Short text.", metadata=meta)]
    settings = _make_settings(chunk_size=1000, chunk_overlap=0)

    chunks = chunk_documents(docs, settings)
    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "test.pdf"
    assert chunks[0].metadata["page"] == 3
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunk_documents_adds_chunk_index():
    text = "sentence one. " * 50
    docs = [Document(page_content=text, metadata={"source": "test.pdf", "page": 0})]
    settings = _make_settings(chunk_size=100, chunk_overlap=10)

    chunks = chunk_documents(docs, settings)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))
