"""Unit tests for backend.rag.retriever, using stubbed embeddings/vector store
so no real model download or Chroma instance is required.
"""

from backend.rag.retriever import Retriever


class _FakeEmbeddingService:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class _FakeVectorStore:
    """Returns a fixed set of candidate vectors/documents for any query."""

    def query(self, query_embedding, top_k=5, where=None, fetch_k=None):
        # 3 candidates: one identical to query, two increasingly different.
        return {
            "ids": [["a", "b", "c"]],
            "documents": [["doc A text", "doc B text", "doc C text"]],
            "metadatas": [[{"filename": "a.txt"}, {"filename": "b.txt"}, {"filename": "c.txt"}]],
            "distances": [[0.0, 0.4, 1.2]],
            "embeddings": [[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]],
        }


def test_similarity_search_orders_by_score():
    retriever = Retriever(
        vector_store=_FakeVectorStore(),
        embedding_service=_FakeEmbeddingService(),
        default_top_k=3,
        default_search_type="similarity",
    )
    sources = retriever.retrieve("test query", search_type="similarity")

    assert len(sources) == 3
    scores = [s.score for s in sources]
    assert scores == sorted(scores, reverse=True)


def test_mmr_search_returns_requested_count():
    retriever = Retriever(
        vector_store=_FakeVectorStore(),
        embedding_service=_FakeEmbeddingService(),
        default_top_k=2,
        default_search_type="mmr",
        mmr_fetch_k=3,
        mmr_lambda=0.5,
    )
    sources = retriever.retrieve("test query", top_k=2, search_type="mmr")

    assert len(sources) == 2
    assert all(0.0 <= s.score <= 1.0 for s in sources)


def test_retrieve_includes_document_metadata():
    retriever = Retriever(
        vector_store=_FakeVectorStore(),
        embedding_service=_FakeEmbeddingService(),
        default_top_k=1,
        default_search_type="similarity",
    )
    sources = retriever.retrieve("test query", top_k=1)
    assert sources[0].document_name == "a.txt"
