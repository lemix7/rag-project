from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.api.dependencies import get_chain, get_vs
from rag.api.middleware import verify_api_key
from rag.generation.chain import query_rag, stream_rag
from rag.observability.logging import get_logger
from rag.retrieval.retriever import search

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)
    search_type: str = Field(default="similarity", pattern="^(similarity|mmr)$")
    stream: bool = Field(default=False)


class SourceDoc(BaseModel):
    content: str
    page: int | str
    source: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        chain = get_chain()
        vector_store = get_vs()

        # Get answer
        answer = query_rag(chain, request.question)

        # Get source documents
        results = search(
            vector_store,
            [request.question],
            k=request.k,
            search_type=request.search_type,
        )

        sources = []
        for doc in results[0] if results else []:
            sources.append(SourceDoc(
                content=doc.page_content[:500],
                page=doc.metadata.get("page", "?"),
                source=doc.metadata.get("source", "unknown"),
            ))

        logger.info(
            "query_answered",
            question_length=len(request.question),
            sources=len(sources),
        )

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    try:
        chain = get_chain()

        async def generate():
            async for chunk in stream_rag(chain, request.question):
                yield chunk

        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        logger.error("stream_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
