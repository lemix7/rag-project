import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from rag.api.dependencies import get_settings, get_vs
from rag.api.middleware import verify_api_key
from rag.ingestion.chunker import chunk_documents
from rag.ingestion.dedup import deduplicate_chunks
from rag.ingestion.loader import SUPPORTED_LOADERS, load_document
from rag.observability.logging import get_logger
from rag.retrieval.vector_store import index_documents

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    settings = get_settings()

    # Validate file size
    contents = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB",
        )

    # Validate file type
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_LOADERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {suffix}. "
                f"Supported: {', '.join(SUPPORTED_LOADERS.keys())}"
            ),
        )

    # Save to temp file and process
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        docs = load_document(tmp_path)
        chunks = chunk_documents(docs, settings)

        # Add original filename to metadata
        for chunk in chunks:
            chunk.metadata["source"] = file.filename

        unique_chunks, chunk_ids = deduplicate_chunks(chunks)
        vector_store = get_vs()
        ids = index_documents(vector_store, unique_chunks, ids=chunk_ids)

        logger.info(
            "document_uploaded",
            filename=file.filename,
            chunks=len(chunks),
            indexed=len(ids),
            duplicates_skipped=len(chunks) - len(unique_chunks),
        )

        return {
            "filename": file.filename,
            "total_chunks": len(chunks),
            "indexed_chunks": len(ids),
            "duplicates_skipped": len(chunks) - len(unique_chunks),
        }
    except Exception as e:
        logger.error("upload_failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/documents/supported-types")
async def supported_types():
    return {"supported_types": list(SUPPORTED_LOADERS.keys())}
