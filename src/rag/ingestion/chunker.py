from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.core.config import Settings
from rag.core.exceptions import ChunkingError
from rag.observability.logging import get_logger

logger = get_logger(__name__)

PDF_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_documents(
    docs: list[Document], settings: Settings
) -> list[Document]:
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=PDF_SEPARATORS,
            add_start_index=True,
        )
        chunks = splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        logger.info(
            "documents_chunked",
            input_docs=len(docs),
            output_chunks=len(chunks),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        return chunks
    except Exception as e:
        raise ChunkingError("Failed to chunk documents", detail=str(e)) from e
