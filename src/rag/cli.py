from dotenv import load_dotenv

from rag.core.config import get_settings
from rag.ingestion.chunker import chunk_documents
from rag.ingestion.dedup import deduplicate_chunks
from rag.ingestion.loader import load_document
from rag.observability.logging import get_logger, setup_logging
from rag.retrieval.retriever import search
from rag.retrieval.vector_store import get_vector_store, index_documents


def main():
    load_dotenv()
    settings = get_settings()
    setup_logging(settings)
    logger = get_logger(__name__)

    file_path = "./data/nike.pdf"
    queries = [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ]

    # Step 1: Load and chunk
    logger.info("loading_document", file=file_path)
    docs = load_document(file_path)
    chunks = chunk_documents(docs, settings)
    print(f"Loaded {len(docs)} pages, split into {len(chunks)} chunks")

    # Step 2: Deduplicate and index
    unique_chunks, chunk_ids = deduplicate_chunks(chunks)
    vector_store = get_vector_store(settings)
    ids = index_documents(vector_store, unique_chunks, ids=chunk_ids)
    skipped = len(chunks) - len(unique_chunks)
    print(f"Indexed {len(ids)} new chunks ({skipped} duplicates skipped)")

    # Step 3: Search
    print("\nRunning semantic search queries:")
    results = search(vector_store, queries, k=1)

    for query, docs in zip(queries, results):
        print(f"\nQ: {query}")
        for doc in docs:
            page = doc.metadata.get("page", "?")
            print(f"  [page {page}] {doc.page_content[:500]}...")


if __name__ == "__main__":
    main()
