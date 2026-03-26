from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()

# Split documents into smaller chunks
def split_documents(docs: List[Document],chunk_size: int = 1000,chunk_overlap: int = 200,) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return text_splitter.split_documents(docs)


def load_and_split(file_path: str) -> List[Document]:

    docs = load_pdf(file_path)
    return split_documents(docs)
