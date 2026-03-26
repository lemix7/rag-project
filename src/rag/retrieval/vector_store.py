from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings

from rag.core.config import Settings
from rag.core.exceptions import EmbeddingError, VectorStoreError
from rag.observability.logging import get_logger

logger = get_logger(__name__)


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    try:
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    except Exception as e:
        raise EmbeddingError("Failed to initialize embeddings", detail=str(e)) from e


def get_vector_store(settings: Settings) -> VectorStore:
    embeddings = get_embeddings(settings)

    try:
        if settings.vector_store_type == "pgvector":
            from langchain_postgres import PGVector

            return PGVector(
                connection=settings.postgres_dsn,
                embeddings=embeddings,
                collection_name=settings.chroma_collection,
            )
        else:
            return Chroma(
                collection_name=settings.chroma_collection,
                embedding_function=embeddings,
                persist_directory=settings.chroma_persist_dir,
            )
    except Exception as e:
        raise VectorStoreError(
            "Failed to initialize vector store", detail=str(e)
        ) from e


def index_documents(
    vector_store: VectorStore,
    documents: list[Document],
    ids: list[str] | None = None,
) -> list[str]:
    try:
        result_ids = vector_store.add_documents(documents, ids=ids)
        logger.info("documents_indexed", count=len(result_ids))
        return result_ids
    except Exception as e:
        raise VectorStoreError("Failed to index documents", detail=str(e)) from e
