from langchain_core.vectorstores import VectorStore

from rag.core.config import Settings
from rag.generation.chain import build_rag_chain

_settings: Settings | None = None
_vector_store: VectorStore | None = None
_rag_chain = None


def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_vs() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        from rag.retrieval.vector_store import get_vector_store as _get_vs
        _vector_store = _get_vs(get_settings())
    return _vector_store


def get_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = build_rag_chain(get_vs(), get_settings())
    return _rag_chain


def reset_dependencies():
    global _settings, _vector_store, _rag_chain
    _settings = None
    _vector_store = None
    _rag_chain = None
