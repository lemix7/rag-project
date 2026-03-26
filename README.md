# RAG Demo Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that loads a PDF document, splits it into chunks, indexes the chunks into a ChromaDB vector store using OpenAI embeddings, and performs similarity search.

## How It Works

The pipeline processes documents in three stages:

1. **Document Loading** (`document_loader.py`) — Reads a PDF file and splits the text into overlapping chunks of 1000 characters (with 200 character overlap) using LangChain's `RecursiveCharacterTextSplitter`.

2. **Indexing** (`vector_store.py`) — Converts each chunk into a vector embedding using OpenAI's `text-embedding-3-large` model and stores them in a local ChromaDB database (`./chroma_langchain_db/`).

3. **Retrieval** (`retriver.py`) — Takes a list of natural language queries and finds the most semantically similar chunks from the vector store.

```
PDF  -->  Chunks  -->  Embeddings  -->  ChromaDB  -->  Similarity Search
```

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- An OpenAI API key

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/lemix7/rag-project.git
cd rag-project
```

### 2. Install dependencies

```bash
uv sync
```

This installs all dependencies defined in `pyproject.toml`:
- `langchain-community` — PDF document loader
- `langchain-text-splitters` — Text chunking
- `langchain-openai` — OpenAI embeddings
- `langchain-chroma` — ChromaDB vector store
- `pypdf` — PDF parsing
- `python-dotenv` — Environment variable management

### 3. Set up your OpenAI API key

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Replace `your-api-key-here` with your actual [OpenAI API key](https://platform.openai.com/api-keys).

### 4. Add a PDF document

The project comes with a sample Nike PDF in `data/nike.pdf`. To use your own document, place a PDF file in the `data/` directory and update the `FILE_PATH` variable in `main.py`.

### 5. Run the pipeline

```bash
uv run python main.py
```

### Expected Output

```
Loading and splitting: ./data/nike.pdf
  → 107 chunks created
Indexing documents into vector store...
  → 107 documents indexed

Running semantic search queries:

Q: How many distribution centers does Nike have in the US?
  [page 21] Nike has distribution centers in Memphis, Tennessee...

Q: When was Nike incorporated?
  [page 1] NIKE, Inc. was incorporated in 1967 under the laws of the State of Oregon...
```

## Project Structure

```
rag-project/
├── main.py               # Pipeline orchestration
├── document_loader.py     # PDF loading and text chunking
├── vector_store.py        # Embedding and ChromaDB management
├── retriver.py            # Similarity search retrieval
├── data/
│   └── nike.pdf           # Sample PDF document
├── chroma_langchain_db/   # Persistent vector store (auto-generated)
├── pyproject.toml         # Project dependencies
└── uv.lock                # Locked dependency versions
```

## Notes

- On the first run, the pipeline embeds all chunks and stores them in ChromaDB. Subsequent runs reuse the persisted database, so re-embedding is skipped unless the database is deleted.
- The default embedding model (`text-embedding-3-large`) provides high-quality embeddings but costs per API call. See [OpenAI pricing](https://openai.com/pricing) for details.
- To reset the vector store, delete the `chroma_langchain_db/` directory and re-run the pipeline.
