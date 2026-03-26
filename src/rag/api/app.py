from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag.api.routes import documents, health, query
from rag.core.exceptions import RAGBaseError
from rag.observability.logging import get_logger, setup_logging
from rag.observability.metrics import setup_metrics


def _make_lifespan(settings):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings)
        logger = get_logger(__name__)
        logger.info("app_starting", vector_store=settings.vector_store_type)
        yield
        logger.info("app_shutdown")

    return lifespan


def create_app(settings=None) -> FastAPI:
    if settings is None:
        load_dotenv()
        from rag.api.dependencies import get_settings
        settings = get_settings()

    app = FastAPI(
        title="RAG API",
        version="0.2.0",
        description="Production-ready RAG pipeline API",
        lifespan=_make_lifespan(settings),
    )

    # Rate limiting
    limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Exception handler for RAG errors
    @app.exception_handler(RAGBaseError)
    async def rag_error_handler(request, exc: RAGBaseError):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"error": exc.message, "detail": exc.detail},
        )

    # Metrics
    setup_metrics(app)

    # Routes
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
    app.include_router(query.router, prefix="/api/v1", tags=["query"])

    return app


def run_server():
    import uvicorn

    load_dotenv()
    from rag.api.dependencies import get_settings
    settings = get_settings()
    uvicorn.run(
        "rag.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
