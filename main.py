from dotenv import load_dotenv

from document_loader import load_and_split
from vector_store import get_vector_store, index_documents
from retriver import search

load_dotenv()

FILE_PATH = "./data/nike.pdf"

QUERIES = [
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?",
]


def main():
    # Step 1: Load and chunk the PDF
    print(f"Loading and splitting: {FILE_PATH}")
    chunks = load_and_split(FILE_PATH)
    print(f"  → {len(chunks)} chunks created")

    # Step 2: Set up vector store and index documents
    print("Indexing documents into vector store...")
    vector_store = get_vector_store()
    ids = index_documents(vector_store, chunks)
    print(f"  → {len(ids)} documents indexed")

    # Step 3: Run semantic search
    print("\nRunning semantic search queries:")
    results = search(vector_store, QUERIES, k=1)

    for query, docs in zip(QUERIES, results):
        print(f"\nQ: {query}")
        for doc in docs:
            print(f"  [page {doc.metadata.get('page', '?')}] {doc.page_content[:500]}...")


if __name__ == "__main__":
    main()