from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based on the provided context.

Rules:
- Only use information from the provided context to answer the question.
- If the context does not contain enough information, say \
"I don't have enough information to answer this question."
- Cite the source page numbers when possible using [page X] format.
- Be concise and direct in your answers.
- Do not make up information or speculate beyond what the context provides."""

RAG_USER_PROMPT = """\
Context:
{context}

Question: {question}"""


def get_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_USER_PROMPT),
    ])
