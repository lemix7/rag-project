from langchain_core.documents import Document

from rag.ingestion.dedup import deduplicate_chunks, generate_chunk_id


def _doc(content: str, source: str = "test.pdf", page: int = 0) -> Document:
    return Document(page_content=content, metadata={"source": source, "page": page})


def test_same_content_produces_same_id():
    doc1 = _doc("Hello world")
    doc2 = _doc("Hello world")
    assert generate_chunk_id(doc1) == generate_chunk_id(doc2)


def test_different_content_produces_different_id():
    doc1 = _doc("Hello world")
    doc2 = _doc("Goodbye world")
    assert generate_chunk_id(doc1) != generate_chunk_id(doc2)


def test_whitespace_normalization():
    doc1 = _doc("Hello   world")
    doc2 = _doc("Hello world")
    assert generate_chunk_id(doc1) == generate_chunk_id(doc2)


def test_different_pages_produce_different_ids():
    doc1 = _doc("Same content", page=1)
    doc2 = _doc("Same content", page=2)
    assert generate_chunk_id(doc1) != generate_chunk_id(doc2)


def test_deduplicate_removes_duplicates():
    chunks = [_doc("aaa"), _doc("bbb"), _doc("aaa")]
    unique, ids = deduplicate_chunks(chunks)
    assert len(unique) == 2
    assert len(ids) == 2


def test_deduplicate_with_existing_ids():
    chunks = [_doc("aaa"), _doc("bbb"), _doc("ccc")]
    existing = {generate_chunk_id(_doc("aaa"))}
    unique, ids = deduplicate_chunks(chunks, existing_ids=existing)
    assert len(unique) == 2
    assert all(uid not in existing for uid in ids)
