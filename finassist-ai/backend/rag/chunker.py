"""Chunking strategies for splitting document text prior to embedding.

Two strategies are supported:

* ``RecursiveChunker`` - wraps LangChain's RecursiveCharacterTextSplitter,
  splitting on paragraph/sentence/word boundaries with configurable
  size and overlap.
* ``SemanticChunker`` - groups sentences using embedding similarity so
  that each chunk stays topically coherent, falling back to recursive
  splitting for any chunk that ends up too large.
"""

from dataclasses import dataclass
from typing import Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A single chunk of text ready for embedding."""

    text: str
    chunk_index: int
    metadata: dict


class Chunker(Protocol):
    """Common interface implemented by all chunking strategies."""

    def split(self, text: str, base_metadata: dict) -> list[Chunk]:
        ...


class RecursiveChunker:
    """Splits text recursively on paragraph -> sentence -> word boundaries."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, text: str, base_metadata: dict) -> list[Chunk]:
        pieces = self._splitter.split_text(text)
        logger.info("Recursive chunking produced %s chunks", len(pieces))
        return [
            Chunk(text=piece, chunk_index=i, metadata=dict(base_metadata))
            for i, piece in enumerate(pieces)
        ]


class SemanticChunker:
    """Groups sentences by embedding-similarity into topically coherent chunks.

    Falls back to a RecursiveChunker pass on any resulting group that
    exceeds ``max_chunk_size`` characters.
    """

    def __init__(
        self,
        embedding_fn,
        similarity_threshold: float = 0.55,
        max_chunk_size: int = 1200,
        chunk_overlap: int = 120,
    ) -> None:
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self._fallback = RecursiveChunker(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)

    def split(self, text: str, base_metadata: dict) -> list[Chunk]:
        import numpy as np

        sentences = [s.strip() for s in _split_sentences(text) if s.strip()]
        if not sentences:
            return []

        embeddings = self.embedding_fn(sentences)
        groups: list[list[str]] = [[sentences[0]]]
        group_vectors = [embeddings[0]]

        for sentence, vector in zip(sentences[1:], embeddings[1:]):
            sim = _cosine_similarity(group_vectors[-1], vector)
            if sim >= self.similarity_threshold:
                groups[-1].append(sentence)
            else:
                groups.append([sentence])
            group_vectors.append(vector)

        chunks: list[Chunk] = []
        idx = 0
        for group in groups:
            joined = " ".join(group)
            if len(joined) > self.max_chunk_size:
                for sub in self._fallback.split(joined, base_metadata):
                    chunks.append(Chunk(text=sub.text, chunk_index=idx, metadata=dict(base_metadata)))
                    idx += 1
            else:
                chunks.append(Chunk(text=joined, chunk_index=idx, metadata=dict(base_metadata)))
                idx += 1

        logger.info("Semantic chunking produced %s chunks", len(chunks))
        return chunks


def _split_sentences(text: str) -> list[str]:
    import re

    sentence_end = r"(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Corp|Ltd|Co|vs|etc|approx|dept|est|govt))[.!?](?:\s+|\Z)"
    return re.split(sentence_end, text)


def _cosine_similarity(a, b) -> float:
    import numpy as np

    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)


def get_chunker(strategy: str, chunk_size: int, chunk_overlap: int, embedding_fn=None) -> Chunker:
    """Factory returning the configured chunking strategy.

    Args:
        strategy: Either "recursive" or "semantic".
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.
        embedding_fn: Required for "semantic" strategy; a callable mapping
            list[str] -> list[vector].

    Returns:
        A Chunker instance.
    """
    if strategy == "semantic":
        if embedding_fn is None:
            raise ValueError("embedding_fn is required for semantic chunking")
        return SemanticChunker(embedding_fn=embedding_fn, max_chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
