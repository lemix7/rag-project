class RAGBaseError(Exception):
    """Base exception for the RAG application."""

    def __init__(self, message: str = "", detail: str = ""):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class DocumentLoadError(RAGBaseError):
    """Raised when a document cannot be loaded or parsed."""


class UnsupportedFileTypeError(RAGBaseError):
    """Raised when a file type is not supported."""


class ChunkingError(RAGBaseError):
    """Raised when document chunking fails."""


class EmbeddingError(RAGBaseError):
    """Raised when embedding generation fails."""


class VectorStoreError(RAGBaseError):
    """Raised when vector store operations fail."""


class GenerationError(RAGBaseError):
    """Raised when LLM generation fails."""


class AuthenticationError(RAGBaseError):
    """Raised when API authentication fails."""


class RateLimitError(RAGBaseError):
    """Raised when rate limit is exceeded."""
