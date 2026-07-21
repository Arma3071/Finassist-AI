"""FastAPI route definitions for FinAssist AI."""

import shutil
import time
from pathlib import Path

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from backend.agent.graph import FinAssistAgent
from backend.config import get_settings
from backend.database import db
from backend.evaluation.dataset import EVAL_DATASET
from backend.evaluation.evaluator import Evaluator
from backend.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentDeleteRequest,
    EvaluationRunResponse,
    EvaluationSummary,
    LoginRequest,
    LoginResponse,
    MetricsResponse,
    RegisterRequest,
    UploadResponse,
)
from backend.rag.loader import SUPPORTED_EXTENSIONS
from backend.rag.pipeline import RAGPipeline
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

_settings = get_settings()
db.init_db()

_rag_pipeline = RAGPipeline(_settings)
_agent = FinAssistAgent(_settings, rag_pipeline=_rag_pipeline)
_evaluator = Evaluator(embedding_service=_rag_pipeline.embedding_service)

_UPLOAD_DIR = Path("./data/uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _current_username(authorization: str | None) -> str | None:
    """Resolve the calling user from an 'Authorization: Bearer <token>' header, if present."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ").strip()
    return db.get_username_for_token(token)


# --------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------- #
@router.post("/auth/register")
def register(request: RegisterRequest) -> dict:
    """Register a new user account."""
    created = db.create_user(request.username, request.password)
    if not created:
        raise HTTPException(status_code=409, detail="Username already exists")
    return {"status": "created", "username": request.username}


@router.post("/auth/login", response_model=LoginResponse)
def login(request: LoginRequest) -> LoginResponse:
    """Authenticate a user and issue a session token."""
    if not db.verify_user(request.username, request.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = db.create_session(request.username)
    return LoginResponse(token=token, username=request.username)


# --------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------- #
@router.get("/health")
def health() -> dict:
    """Liveness/readiness check."""
    return {
        "status": "ok",
        "vector_count": _rag_pipeline.vector_store.count(),
        "llm_provider": _settings.llm_provider,
    }


# --------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------- #
@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...), authorization: str = Header(...)
) -> UploadResponse:
    """Upload and ingest a document (PDF, DOCX, TXT, or Markdown) into the knowledge base."""
    username = _current_username(authorization)
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token.")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    dest_path = _UPLOAD_DIR / file.filename
    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info("Received upload from user '%s': %s", username, file.filename)
    result = _rag_pipeline.ingest(str(dest_path))
    if result.status == "failed":
        raise HTTPException(status_code=422, detail=result.message)

    db.log_document(
        document_id=result.document_id,
        filename=result.filename,
        chunks_created=result.chunks_created,
        uploaded_by=username,
    )
    return result


@router.delete("/documents")
def delete_document(request: DocumentDeleteRequest) -> dict:
    """Delete a document and all its chunks from the vector store."""
    deleted = _rag_pipeline.delete_document(request.document_id)
    return {"document_id": request.document_id, "chunks_deleted": deleted}


# --------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------- #
@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, authorization: str = Header(...)) -> ChatResponse:
    """Answer a user message via the FinAssist agent (RAG + MCP tools)."""
    username = _current_username(authorization)
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token.")

    start = time.perf_counter()
    history = db.get_chat_history(request.session_id)
    history_for_agent = [{"role": h["role"], "content": h["content"]} for h in history]

    try:
        response = _agent.run(
            question=request.message,
            session_id=request.session_id,
            conversation_history=history_for_agent,
            top_k=request.top_k,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat request failed for session %s", request.session_id)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    db.log_chat_message(request.session_id, "user", request.message, username)
    db.log_chat_message(request.session_id, "assistant", response.answer, username)

    avg_retrieval_score = (
        sum(s.score for s in response.sources) / len(response.sources) if response.sources else None
    )
    db.log_query(
        session_id=request.session_id,
        question=request.message,
        route=response.route,
        latency_ms=response.latency_ms,
        confidence=response.confidence,
        retrieval_score_avg=avg_retrieval_score,
        username=username,
    )

    logger.info(
        "Chat session=%s route=%s latency=%.1fms total=%.1fms",
        request.session_id,
        response.route,
        response.latency_ms,
        (time.perf_counter() - start) * 1000,
    )
    return response


@router.get("/history/{session_id}")
def get_history(session_id: str) -> dict:
    """Return the stored conversation history for a session."""
    return {"session_id": session_id, "history": db.get_chat_history(session_id)}


# --------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------- #
@router.post("/evaluate", response_model=EvaluationRunResponse)
def run_evaluation() -> EvaluationRunResponse:
    """Run the golden evaluation dataset through the agent and score the results."""
    results = _evaluator.evaluate_dataset(_agent, EVAL_DATASET)

    summaries = [
        EvaluationSummary(
            question=r.question,
            retrieval_precision=r.retrieval_precision,
            retrieval_recall=r.retrieval_recall,
            latency_ms=r.latency_ms,
            hallucination_rate=r.hallucination_rate,
            context_relevance=r.context_relevance,
            answer_relevance=r.answer_relevance,
        )
        for r in results
    ]

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    aggregate = {
        "avg_precision": avg([r.retrieval_precision for r in results]),
        "avg_recall": avg([r.retrieval_recall for r in results]),
        "avg_latency_ms": avg([r.latency_ms for r in results]),
        "avg_hallucination_rate": avg([r.hallucination_rate for r in results]),
        "avg_context_relevance": avg([r.context_relevance for r in results]),
        "avg_answer_relevance": avg([r.answer_relevance for r in results]),
    }

    return EvaluationRunResponse(items_evaluated=len(results), results=summaries, aggregate=aggregate)


# --------------------------------------------------------------------- #
# Metrics / Admin dashboard
# --------------------------------------------------------------------- #
@router.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """Return aggregated metrics for the admin dashboard."""
    query_stats = db.get_query_stats()
    eval_stats = db.get_evaluation_stats()

    return MetricsResponse(
        documents_uploaded=db.count_documents(),
        embedding_count=_rag_pipeline.vector_store.count(),
        total_queries=query_stats["total_queries"],
        avg_latency_ms=round(query_stats["avg_latency_ms"], 1),
        avg_confidence=round(query_stats["avg_confidence"], 3),
        avg_retrieval_score=round(query_stats["avg_retrieval_score"], 3),
        queries_by_route=query_stats["by_route"],
        evaluation={
            "total_evaluations": eval_stats["total_evaluations"],
            "avg_precision": round(eval_stats["avg_precision"], 3),
            "avg_recall": round(eval_stats["avg_recall"], 3),
            "avg_hallucination_rate": round(eval_stats["avg_hallucination_rate"], 3),
            "avg_context_relevance": round(eval_stats["avg_context_relevance"], 3),
            "avg_answer_relevance": round(eval_stats["avg_answer_relevance"], 3),
        },
    )
