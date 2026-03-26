from fastapi import APIRouter

from rag.api.dependencies import get_settings, get_vs

router = APIRouter()


@router.get("/health")
async def health_check():
    settings = get_settings()
    checks = {"status": "healthy", "vector_store": settings.vector_store_type}

    try:
        vs = get_vs()
        # Quick connectivity test
        vs.similarity_search("test", k=1)
        checks["vector_store_connected"] = True
    except Exception:
        checks["vector_store_connected"] = False
        checks["status"] = "degraded"

    return checks
