"""Unit tests for backend.evaluation.evaluator.Evaluator, fully stubbed
(no real LLM calls, embedding model, or vector store required).
"""

from backend.database import db
from backend.evaluation.evaluator import Evaluator
from backend.models.schemas import ChatResponse, Source


class _FakeEmbeddingService:
    """Deterministic fake embeddings: identical strings -> identical vectors."""

    def embed_query(self, text: str) -> list[float]:
        # crude bag-of-words style vector so relevance isn't trivially 0 or 1
        return [float(len(text)), float(text.count("revenue")), float(text.count("margin"))]


class _FakeAgent:
    def __init__(self, response: ChatResponse):
        self._response = response

    def run(self, question: str, session_id: str):
        return self._response


def test_evaluate_item_computes_all_metrics():
    db.init_db()
    response = ChatResponse(
        session_id="eval",
        answer="Revenue grew 14% to $482 million [S1].",
        confidence=0.8,
        sources=[Source(chunk_id="1", document_name="sample_report.md", text="...", score=0.85)],
        route="retrieval",
        latency_ms=50.0,
    )
    agent = _FakeAgent(response)
    evaluator = Evaluator(embedding_service=_FakeEmbeddingService())

    item = {
        "question": "What was Q3 revenue growth?",
        "relevant_documents": ["sample_report.md"],
        "expected_keywords": ["revenue", "482"],
    }
    result = evaluator.evaluate_item(agent, item)

    assert result.retrieval_precision == 1.0
    assert result.retrieval_recall == 1.0
    assert 0.0 <= result.hallucination_rate <= 1.0
    assert result.context_relevance == 0.85
    assert 0.0 <= result.answer_relevance <= 1.0


def test_evaluate_item_precision_recall_when_wrong_doc_retrieved():
    db.init_db()
    response = ChatResponse(
        session_id="eval",
        answer="I don't have enough information to answer that.",
        confidence=0.2,
        sources=[Source(chunk_id="1", document_name="unrelated.pdf", text="...", score=0.4)],
        route="retrieval",
        latency_ms=40.0,
    )
    agent = _FakeAgent(response)
    evaluator = Evaluator(embedding_service=_FakeEmbeddingService())

    item = {"question": "What was Q3 revenue?", "relevant_documents": ["sample_report.md"]}
    result = evaluator.evaluate_item(agent, item)

    assert result.retrieval_precision == 0.0
    assert result.retrieval_recall == 0.0


def test_hallucination_rate_penalizes_uncited_claims():
    evaluator = Evaluator(embedding_service=_FakeEmbeddingService())
    fully_cited = "Revenue grew 14% to $482 million in Q3 [S1]. Margin improved to 61.2% [S2]."
    uncited = "Revenue grew 14% to $482 million in Q3. Margin improved to 61.2% as well."

    assert evaluator._hallucination_rate(fully_cited) == 0.0
    assert evaluator._hallucination_rate(uncited) == 1.0
