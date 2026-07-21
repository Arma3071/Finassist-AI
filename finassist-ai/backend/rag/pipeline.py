"""End-to-end RAG pipeline: ingestion and query-answering.

Ingestion:  loader -> chunker -> embeddings -> vector store
Querying:   retriever -> prompt builder -> LLM -> answer + sources + confidence
"""

import time
import uuid
from pathlib import Path

from backend.config import Settings
from backend.llm.client import LLMClient
from backend.models.schemas import ChatResponse, Source, UploadResponse
from backend.rag.chunker import get_chunker
from backend.rag.embeddings import get_embedding_service
from backend.rag.loader import DocumentLoader
from backend.rag.prompt import PromptBuilder
from backend.rag.retriever import Retriever
from backend.rag.vectorstore import VectorStore
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Coordinates document ingestion and retrieval-augmented question answering."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loader = DocumentLoader()
        self.embedding_service = get_embedding_service(model_name=settings.embedding_model)
        self.vector_store = VectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            default_top_k=settings.retrieval_top_k,
            default_search_type=settings.retrieval_search_type,
            mmr_fetch_k=settings.mmr_fetch_k,
            mmr_lambda=settings.mmr_lambda,
        )
        self.prompt_builder = PromptBuilder()
        self.llm_client = LLMClient(settings)

    def ingest(self, file_path: str) -> UploadResponse:
        """Ingest a single document: load, clean, chunk, embed, store.

        Args:
            file_path: Path to the uploaded file on disk.

        Returns:
            An UploadResponse describing the outcome.
        """
        document_id = uuid.uuid4().hex
        try:
            doc = self.loader.load(file_path)
            base_metadata = {
                "filename": doc.filename,
                "extension": doc.extension,
                "headings": ", ".join(doc.headings[:10]),
            }

            embed_fn = self.embedding_service if self.settings.chunking_strategy == "semantic" else None
            chunker = get_chunker(
                strategy=self.settings.chunking_strategy,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                embedding_fn=embed_fn,
            )
            chunks = chunker.split(doc.text, base_metadata)
            if not chunks:
                return UploadResponse(
                    document_id=document_id,
                    filename=doc.filename,
                    chunks_created=0,
                    status="failed",
                    message="No extractable text found in document.",
                )

            embeddings = self.embedding_service.embed_documents([c.text for c in chunks])
            self.vector_store.add_chunks(chunks, embeddings, document_id)

            return UploadResponse(
                document_id=document_id,
                filename=doc.filename,
                chunks_created=len(chunks),
                status="success",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingestion failed for %s", file_path)
            return UploadResponse(
                document_id=document_id,
                filename=Path(file_path).name,
                chunks_created=0,
                status="failed",
                message=str(exc),
            )

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: The document identifier.

        Returns:
            Number of chunks deleted.
        """
        return self.vector_store.delete_document(document_id)

    def answer(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
        tool_results: list[dict] | None = None,
    ) -> ChatResponse:
        """Answer a question using retrieval-augmented generation.

        Args:
            question: The user's question.
            conversation_history: Prior turns for context.
            top_k: Override for number of chunks to retrieve.
            metadata_filter: Optional metadata filter for retrieval.
            tool_results: Optional MCP tool results to fold into the prompt.

        Returns:
            A populated ChatResponse with answer, confidence, and sources.
        """
        start = time.perf_counter()
        sources = self.retriever.retrieve(question, top_k=top_k, metadata_filter=metadata_filter)

        prompt = self.prompt_builder.build(
            question=question,
            sources=sources,
            conversation_history=conversation_history,
            tool_results=tool_results,
        )
        answer_text = self.llm_client.complete(prompt)
        confidence = self._estimate_confidence(sources, top_k=top_k or self.settings.retrieval_top_k)
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info("Answered question in %.1fms (confidence=%.2f)", latency_ms, confidence)

        return ChatResponse(
            session_id="",  # populated by the caller/agent
            answer=answer_text,
            confidence=confidence,
            sources=sources,
            route="retrieval",
            latency_ms=latency_ms,
        )

    @staticmethod
    def _estimate_confidence(sources: list[Source], top_k: int = 5) -> float:
        """Estimate answer confidence from retrieval scores."""
        if not sources:
            return 0.15
        avg_score = sum(s.score for s in sources) / len(sources)
        expected_min = max(1, int(top_k * 0.6))
        coverage_penalty = min(1.0, len(sources) / expected_min)
        return round(max(0.0, min(1.0, avg_score * coverage_penalty)), 2)
