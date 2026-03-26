from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from rag.core.exceptions import DocumentLoadError, UnsupportedFileTypeError
from rag.observability.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_document(file_path: str) -> list[Document]:
    path = Path(file_path)

    if not path.exists():
        raise DocumentLoadError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    loader_cls = SUPPORTED_LOADERS.get(suffix)

    if loader_cls is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {suffix}",
            detail=f"Supported types: {', '.join(SUPPORTED_LOADERS.keys())}",
        )

    try:
        loader = loader_cls(str(path))
        docs = loader.load()
        logger.info("document_loaded", file=file_path, pages=len(docs))
        return docs
    except Exception as e:
        raise DocumentLoadError(
            f"Failed to load {file_path}", detail=str(e)
        ) from e
