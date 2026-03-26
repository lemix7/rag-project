from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from rag.observability.logging import get_logger

logger = get_logger(__name__)


def get_retriever(
    vector_store: VectorStore,
    k: int = 4,
    search_type: str = "similarity",
    fetch_k: int = 20,
    score_threshold: float | None = None,
):
    search_kwargs = {"k": k}

    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k
    elif search_type == "similarity_score_threshold" and score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def search(
    vector_store: VectorStore,
    queries: list[str],
    k: int = 4,
    search_type: str = "similarity",
) -> list[list[Document]]:
    retriever = get_retriever(vector_store, k=k, search_type=search_type)
    results = retriever.batch(queries)
    logger.info("search_completed", queries=len(queries), search_type=search_type, k=k)
    return results
