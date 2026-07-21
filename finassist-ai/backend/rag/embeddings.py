"""Embedding generation via SentenceTransformers.

The model is configurable (default ``BAAI/bge-base-en-v1.5``). A single
instance should be reused across the app since loading the model is
expensive; :func:`get_embedding_service` provides a cached singleton per
model name.
"""

from functools import lru_cache

from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Generates dense vector embeddings for text using SentenceTransformers."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", model_name)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks.

        Args:
            texts: List of chunk texts.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: The query text.

        Returns:
            The embedding vector.
        """
        vector = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return vector[0].tolist()

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Allow the service to be used as a plain embedding_fn(texts) callable."""
        return self.embed_documents(texts)


@lru_cache
def get_embedding_service(model_name: str = "BAAI/bge-base-en-v1.5") -> EmbeddingService:
    """Return a cached EmbeddingService for the given model name."""
    return EmbeddingService(model_name=model_name)
