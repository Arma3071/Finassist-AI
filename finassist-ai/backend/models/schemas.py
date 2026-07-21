"""Shared Pydantic models for API requests/responses and internal data."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single retrieved source chunk cited in an answer."""

    chunk_id: str
    document_name: str
    text: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """Record of an MCP tool invocation made by the agent."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any
    success: bool
    latency_ms: float
    error: str | None = None


class ChatRequest(BaseModel):
    """Incoming chat message from the user."""

    session_id: str = Field(description="Conversation/session identifier.")
    message: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0)
    metadata_filter: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    """Full response returned by the /chat endpoint."""

    session_id: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[Source] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    route: Literal["retrieval", "tool", "direct", "hybrid"] = "direct"
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UploadResponse(BaseModel):
    """Result of a document upload/ingestion request."""

    document_id: str
    filename: str
    chunks_created: int
    status: Literal["success", "failed"]
    message: str = ""


class DocumentDeleteRequest(BaseModel):
    """Request body for deleting a document from the vector store."""

    document_id: str


class RegisterRequest(BaseModel):
    """Request body for user registration."""

    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    """Request body for user login."""

    username: str
    password: str


class LoginResponse(BaseModel):
    """Response returned after a successful login."""

    token: str
    username: str


class EvaluationSummary(BaseModel):
    """A single evaluated question's metrics."""

    question: str
    retrieval_precision: float
    retrieval_recall: float
    latency_ms: float
    hallucination_rate: float
    context_relevance: float
    answer_relevance: float


class EvaluationRunResponse(BaseModel):
    """Response returned after running an evaluation batch."""

    items_evaluated: int
    results: list[EvaluationSummary]
    aggregate: dict[str, float]


class MetricsResponse(BaseModel):
    """Aggregated metrics for the admin dashboard."""

    documents_uploaded: int
    embedding_count: int
    total_queries: int
    avg_latency_ms: float
    avg_confidence: float
    avg_retrieval_score: float
    queries_by_route: dict[str, int]
    evaluation: dict[str, float]
