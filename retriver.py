from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List


# Create a similarity-based retriever from the vector store
def get_retriever(vector_store: Chroma, k: int = 1):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# Run a batch of queries against the vector store and return results
def search(vector_store: Chroma, queries: List[str], k: int = 1):
    retriever = get_retriever(vector_store, k=k)
    return retriever.batch(queries)