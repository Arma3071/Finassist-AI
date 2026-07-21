"""Evaluation framework for FinAssist AI's RAG pipeline and agent.

Computes, per question:

* **Retrieval precision / recall** — against a golden set of documents
  that should have been retrieved.
* **Latency** — wall-clock time for the full answer.
* **Hallucination rate** — proxy metric: fraction of answer sentences
  that make a substantive claim without an inline [S#] citation.
* **Context relevance** — average similarity score of retrieved chunks.
* **Answer relevance** — cosine similarity between the question and the
  generated answer in embedding space (does the answer address the
  question at all).

Results are persisted to SQLite via :mod:`backend.database.db` and can be
aggregated for the admin dashboard.
"""

import re
import time
from dataclasses import dataclass, field

import numpy as np

from backend.database import db
from backend.models.schemas import ChatResponse
from backend.rag.embeddings import EmbeddingService
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

_CITATION_PATTERN = re.compile(r"\[S\d+\]")


@dataclass
class EvaluationResult:
    """Metrics for a single evaluated question."""

    question: str
    answer: str
    retrieval_precision: float
    retrieval_recall: float
    latency_ms: float
    hallucination_rate: float
    context_relevance: float
    answer_relevance: float
    retrieved_documents: list[str] = field(default_factory=list)


class Evaluator:
    """Runs evaluation questions through the agent/pipeline and scores the results."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    def evaluate_item(self, agent, item: dict) -> EvaluationResult:
        """Evaluate a single golden-dataset item end-to-end.

        Args:
            agent: A FinAssistAgent (or any object exposing .run(question, session_id)).
            item: A dict with "question", "relevant_documents", "expected_keywords".

        Returns:
            An EvaluationResult with all computed metrics.
        """
        start = time.perf_counter()
        response: ChatResponse = agent.run(question=item["question"], session_id="eval")
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_docs = [s.document_name for s in response.sources]
        precision, recall = self._precision_recall(retrieved_docs, item.get("relevant_documents", []))
        hallucination_rate = self._hallucination_rate(response.answer)
        context_relevance = self._context_relevance(response.sources)
        answer_relevance = self._answer_relevance(item["question"], response.answer)

        result = EvaluationResult(
            question=item["question"],
            answer=response.answer,
            retrieval_precision=precision,
            retrieval_recall=recall,
            latency_ms=latency_ms,
            hallucination_rate=hallucination_rate,
            context_relevance=context_relevance,
            answer_relevance=answer_relevance,
            retrieved_documents=retrieved_docs,
        )

        db.log_evaluation(
            question=result.question,
            answer=result.answer,
            retrieval_precision=result.retrieval_precision,
            retrieval_recall=result.retrieval_recall,
            latency_ms=result.latency_ms,
            hallucination_rate=result.hallucination_rate,
            context_relevance=result.context_relevance,
            answer_relevance=result.answer_relevance,
        )
        return result

    def evaluate_dataset(self, agent, dataset: list[dict]) -> list[EvaluationResult]:
        """Evaluate every item in a dataset and return all results.

        Args:
            agent: The agent to evaluate.
            dataset: List of golden-dataset items.

        Returns:
            List of EvaluationResult, one per item.
        """
        results = [self.evaluate_item(agent, item) for item in dataset]
        logger.info("Evaluated %s dataset items", len(results))
        return results

    # ------------------------------------------------------------------ #
    # Metric implementations
    # ------------------------------------------------------------------ #
    @staticmethod
    def _precision_recall(retrieved_docs: list[str], relevant_docs: list[str]) -> tuple[float, float]:
        if not retrieved_docs and not relevant_docs:
            return 1.0, 1.0
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        if not retrieved_set:
            return 0.0, 0.0
        true_positives = len(retrieved_set & relevant_set)
        precision = true_positives / len(retrieved_set)
        recall = true_positives / len(relevant_set) if relevant_set else 1.0
        return round(precision, 3), round(recall, 3)

    @staticmethod
    def _hallucination_rate(answer: str) -> float:
        """Proxy: fraction of substantive sentences lacking an inline citation."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 15]
        if not sentences:
            return 0.0
        uncited = [s for s in sentences if not _CITATION_PATTERN.search(s)]
        return round(len(uncited) / len(sentences), 3)

    @staticmethod
    def _context_relevance(sources) -> float:
        if not sources:
            return 0.0
        return round(sum(s.score for s in sources) / len(sources), 3)

    def _answer_relevance(self, question: str, answer: str) -> float:
        if not answer.strip():
            return 0.0
        q_vec = np.array(self.embedding_service.embed_query(question))
        a_vec = np.array(self.embedding_service.embed_query(answer))
        denom = (np.linalg.norm(q_vec) * np.linalg.norm(a_vec)) or 1e-8
        similarity = float(np.dot(q_vec, a_vec) / denom)
        return round(max(0.0, min(1.0, similarity)), 3)
