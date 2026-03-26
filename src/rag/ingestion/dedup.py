import hashlib
import re

from langchain_core.documents import Document

from rag.observability.logging import get_logger

logger = get_logger(__name__)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def generate_chunk_id(chunk: Document) -> str:
    source = chunk.metadata.get("source", "")
    page = str(chunk.metadata.get("page", ""))
    content = _normalize(chunk.page_content)
    raw = f"{source}:{page}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()


def deduplicate_chunks(
    chunks: list[Document], existing_ids: set[str] | None = None
) -> tuple[list[Document], list[str]]:
    existing_ids = existing_ids or set()
    unique_chunks = []
    chunk_ids = []
    seen = set()

    for chunk in chunks:
        chunk_id = generate_chunk_id(chunk)
        if chunk_id not in existing_ids and chunk_id not in seen:
            unique_chunks.append(chunk)
            chunk_ids.append(chunk_id)
            seen.add(chunk_id)

    skipped = len(chunks) - len(unique_chunks)
    if skipped > 0:
        logger.info(
            "chunks_deduplicated",
            total=len(chunks),
            skipped=skipped,
            kept=len(unique_chunks),
        )

    return unique_chunks, chunk_ids
