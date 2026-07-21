"""Shared state passed between LangGraph agent nodes."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State object threaded through the FinAssist agent graph.

    Attributes:
        question: The user's current question.
        session_id: Conversation/session identifier.
        conversation_history: Prior turns as [{"role", "content"}, ...].
        route: Decision made by the router node ("retrieval", "tool", "direct", "hybrid").
        sources: Retrieved RAG sources, if any.
        tool_calls: Structured record of MCP tool calls made.
        tool_results: Raw results from tool calls, formatted for prompting.
        answer: Final generated answer text.
        confidence: Estimated confidence score in [0, 1].
    """

    question: str
    session_id: str
    conversation_history: list[dict[str, str]]
    route: str
    top_k: int | None
    sources: list[Any]
    tool_calls: list[Any]
    tool_results: list[dict]
    answer: str
    confidence: float
