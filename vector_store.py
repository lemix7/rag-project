from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List


def get_embeddings(model: str = "text-embedding-3-large") -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model)


def get_vector_store(collection_name: str = "example_collection", persist_directory: str = "./chroma_langchain_db") :
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def index_documents(vector_store: Chroma, documents: List[Document]):
    return vector_store.add_documents(documents=documents)
