"""Retriever with similarity search, MMR, and metadata filtering."""

from typing import Any

import numpy as np

from backend.models.schemas import Source
from backend.rag.embeddings import EmbeddingService
from backend.rag.vectorstore import VectorStore
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class Retriever:
    """Retrieves relevant chunks from the vector store for a query."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        default_top_k: int = 5,
        default_search_type: str = "mmr",
        mmr_fetch_k: int = 20,
        mmr_lambda: float = 0.5,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.default_top_k = default_top_k
        self.default_search_type = default_search_type
        self.mmr_fetch_k = mmr_fetch_k
        self.mmr_lambda = mmr_lambda

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        search_type: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Source]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: The user's natural-language query.
            top_k: Number of results to return (defaults to configured value).
            search_type: "similarity" or "mmr" (defaults to configured value).
            metadata_filter: Optional Chroma-style ``where`` filter.

        Returns:
            List of Source objects ranked by relevance, each with a
            normalized similarity score in [0, 1].
        """
        top_k = top_k or self.default_top_k
        search_type = search_type or self.default_search_type
        query_vector = self.embedding_service.embed_query(query)

        if search_type == "mmr":
            sources = self._mmr_search(query_vector, top_k, metadata_filter)
        else:
            sources = self._similarity_search(query_vector, top_k, metadata_filter)

        logger.info(
            "Retrieved %s chunks for query (search_type=%s, top_k=%s)",
            len(sources),
            search_type,
            top_k,
        )
        return sources

    def _similarity_search(
        self, query_vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[Source]:
        result = self.vector_store.query(query_vector, top_k=top_k, where=metadata_filter)
        return self._to_sources(result)

    def _mmr_search(
        self, query_vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[Source]:
        result = self.vector_store.query(
            query_vector, top_k=top_k, fetch_k=self.mmr_fetch_k, where=metadata_filter
        )
        docs = result.get("documents", [[]])[0]
        if not docs:
            return []

        embeddings = np.array(result["embeddings"][0])
        query_vec = np.array(query_vector)

        selected_idx = self._mmr_select(query_vec, embeddings, top_k, self.mmr_lambda)
        return self._to_sources(result, indices=selected_idx)

    @staticmethod
    def _mmr_select(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int, lambda_mult: float) -> list[int]:
        """Maximal Marginal Relevance selection over candidate document vectors."""

        def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
            return float(np.dot(a, b) / denom)

        candidate_indices = list(range(len(doc_vecs)))
        relevance = [cos_sim(query_vec, doc_vecs[i]) for i in candidate_indices]

        selected: list[int] = []
        while candidate_indices and len(selected) < k:
            if not selected:
                best = max(candidate_indices, key=lambda i: relevance[i])
            else:
                def mmr_score(i: int) -> float:
                    diversity = max(cos_sim(doc_vecs[i], doc_vecs[j]) for j in selected)
                    return lambda_mult * relevance[i] - (1 - lambda_mult) * diversity

                best = max(candidate_indices, key=mmr_score)
            selected.append(best)
            candidate_indices.remove(best)
        return selected

    @staticmethod
    def _to_sources(result: dict[str, Any], indices: list[int] | None = None) -> list[Source]:
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        idx_range = indices if indices is not None else list(range(len(docs)))
        sources = []
        for i in idx_range:
            # Chroma cosine distance -> similarity score in [0, 1]
            distance = distances[i] if i < len(distances) else 1.0
            score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            metadata = metadatas[i] or {}
            sources.append(
                Source(
                    chunk_id=ids[i],
                    document_name=metadata.get("filename", "unknown"),
                    text=docs[i],
                    score=score,
                    metadata=metadata,
                )
            )
        return sources
