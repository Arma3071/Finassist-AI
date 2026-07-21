"""Persistent ChromaDB vector store wrapper.

Wraps a persistent Chroma collection and exposes add / update / delete /
query operations used by the ingestion pipeline and retriever.
"""

import uuid
from typing import Any

import chromadb

from backend.rag.chunker import Chunk
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            "Connected to Chroma collection '%s' at %s (count=%s)",
            collection_name,
            persist_dir,
            self._collection.count(),
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        document_id: str,
    ) -> list[str]:
        """Add embedded chunks to the collection.

        Args:
            chunks: The text chunks to store.
            embeddings: Parallel list of embedding vectors.
            document_id: Identifier of the parent document, stored in metadata
                so all chunks can later be deleted/updated together.

        Returns:
            The list of generated chunk IDs.
        """
        ids = [f"{document_id}::{c.chunk_index}::{uuid.uuid4().hex[:8]}" for c in chunks]
        metadatas = []
        for chunk in chunks:
            meta = dict(chunk.metadata)
            meta["document_id"] = document_id
            meta["chunk_index"] = chunk.chunk_index
            metadatas.append(meta)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=metadatas,
        )
        logger.info("Added %s chunks for document_id=%s", len(ids), document_id)
        return ids

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document.

        Args:
            document_id: The document identifier to remove.

        Returns:
            Number of chunks deleted (best-effort count).
        """
        existing = self._collection.get(where={"document_id": document_id})
        count = len(existing.get("ids", []))
        if count:
            self._collection.delete(where={"document_id": document_id})
        logger.info("Deleted %s chunks for document_id=%s", count, document_id)
        return count

    def update_document(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Replace all chunks for a document with a new set.

        Args:
            document_id: The document identifier to update.
            chunks: New chunks to store.
            embeddings: Parallel embedding vectors for the new chunks.

        Returns:
            The list of new chunk IDs.
        """
        self.delete_document(document_id)
        return self.add_chunks(chunks, embeddings, document_id)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        fetch_k: int | None = None,
    ) -> dict[str, Any]:
        """Run a raw similarity query against the collection.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to request from Chroma (fetch_k for MMR
                pre-filtering, or top_k directly for plain similarity).
            where: Optional metadata filter, e.g. {"document_id": "abc"}.
            fetch_k: Optional override for how many candidates to fetch.

        Returns:
            Raw Chroma query result dict with ids, documents, metadatas, distances.
        """
        n = fetch_k or top_k
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    def count(self) -> int:
        """Return the total number of chunks stored."""
        return self._collection.count()
