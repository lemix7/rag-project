from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from rag.core.config import Settings
from rag.core.exceptions import GenerationError
from rag.generation.prompts import get_rag_prompt
from rag.observability.logging import get_logger
from rag.retrieval.retriever import get_retriever

logger = get_logger(__name__)


def _format_docs(docs: list[Document]) -> str:
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vector_store: VectorStore, settings: Settings):
    try:
        retriever = get_retriever(vector_store, k=4, search_type="similarity")
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
        )
        prompt = get_rag_prompt()

        chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("rag_chain_built", model=settings.llm_model)
        return chain
    except Exception as e:
        raise GenerationError("Failed to build RAG chain", detail=str(e)) from e


def query_rag(chain, question: str) -> str:
    try:
        result = chain.invoke(question)
        logger.info("rag_query_completed", question_length=len(question))
        return result
    except Exception as e:
        raise GenerationError("RAG query failed", detail=str(e)) from e


async def aquery_rag(chain, question: str) -> str:
    try:
        result = await chain.ainvoke(question)
        logger.info(
            "rag_query_completed",
            question_length=len(question),
            async_mode=True,
        )
        return result
    except Exception as e:
        raise GenerationError("RAG async query failed", detail=str(e)) from e


async def stream_rag(chain, question: str):
    try:
        async for chunk in chain.astream(question):
            yield chunk
    except Exception as e:
        raise GenerationError("RAG streaming failed", detail=str(e)) from e
