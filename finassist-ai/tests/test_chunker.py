"""Unit tests for backend.rag.chunker."""

from backend.rag.chunker import RecursiveChunker, get_chunker


def test_recursive_chunker_respects_size_bounds(sample_text):
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split(sample_text, base_metadata={"filename": "test.txt"})

    assert len(chunks) > 1
    for chunk in chunks:
        # Allow a little slack since the splitter can't always cut exactly at size.
        assert len(chunk.text) <= 140
        assert chunk.metadata["filename"] == "test.txt"


def test_recursive_chunker_preserves_content(sample_text):
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
    chunks = chunker.split(sample_text, base_metadata={})

    reconstructed = " ".join(c.text for c in chunks)
    assert "482 million" in reconstructed
    assert "61.2%" in reconstructed


def test_get_chunker_factory_recursive():
    chunker = get_chunker("recursive", chunk_size=500, chunk_overlap=50)
    assert isinstance(chunker, RecursiveChunker)


def test_empty_text_produces_no_chunks():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.split("", base_metadata={})
    assert chunks == []
