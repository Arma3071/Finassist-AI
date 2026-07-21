"""LangGraph agent for FinAssist AI.

The agent decides, per user turn, whether to:

* retrieve documents from the knowledge base (RAG),
* call one or more MCP tools (stock prices, financials, currency, news,
  calculator, date/time),
* answer directly from the conversation, or
* do a hybrid of retrieval + tool calls,

then synthesizes a final grounded answer with sources and confidence.
"""

import json
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from backend.agent.langchain_tools import build_langchain_tools
from backend.agent.state import AgentState
from backend.config import Settings
from backend.llm.client import LLMClient
from backend.mcp.tools.registry import get_tools_by_name
from backend.models.schemas import ChatResponse, ToolCall
from backend.rag.pipeline import RAGPipeline
from backend.rag.prompt import PromptBuilder
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

_ROUTER_SYSTEM_PROMPT = """You are a routing controller for a financial assistant.
Given the user's question, decide the best route:
- "retrieval": the question needs information from uploaded documents/knowledge base.
- "tool": the question needs a live financial fact (stock price, financial statement,
  currency conversion, news, a calculation, or the current date/time).
- "direct": the question is conversational/general and needs neither.
- "hybrid": the question needs both retrieved documents AND a live tool result.
Respond with ONLY one word: retrieval, tool, direct, or hybrid."""


class FinAssistAgent:
    """LangGraph-based agent that ties together RAG and MCP tools."""

    def __init__(self, settings: Settings, rag_pipeline: RAGPipeline | None = None) -> None:
        self.settings = settings
        self.llm_client = LLMClient(settings)
        self.rag_pipeline = rag_pipeline or RAGPipeline(settings)
        self.prompt_builder = PromptBuilder()
        self.lc_tools = build_langchain_tools()
        self.tool_model = self.llm_client.tool_calling_model().bind_tools(self.lc_tools)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("route", self._route_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("call_tools", self._call_tools_node)
        graph.add_node("generate", self._generate_node)

        graph.set_entry_point("route")
        graph.add_conditional_edges(
            "route",
            self._route_selector,
            {
                "retrieval": "retrieve",
                "tool": "call_tools",
                "hybrid": "retrieve",  # retrieve first, then tools, then generate
                "direct": "generate",
            },
        )
        graph.add_conditional_edges(
            "retrieve",
            lambda s: "call_tools" if s.get("route") == "hybrid" else "generate",
            {"call_tools": "call_tools", "generate": "generate"},
        )
        graph.add_edge("call_tools", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    @staticmethod
    def _route_selector(state: AgentState) -> str:
        return state.get("route", "direct")

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #
    def _route_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        raw = self.llm_client.complete(prompt=question, system=_ROUTER_SYSTEM_PROMPT).strip().lower()
        route = next((r for r in ("retrieval", "tool", "direct", "hybrid") if r in raw), "direct")
        logger.info("Router decided route='%s' for question: %s", route, question[:80])
        return {**state, "route": route}

    def _retrieve_node(self, state: AgentState) -> AgentState:
        top_k = state.get("top_k")
        sources = self.rag_pipeline.retriever.retrieve(state["question"], top_k=top_k)
        return {**state, "sources": sources}

    def _call_tools_node(self, state: AgentState) -> AgentState:
        messages = [
            SystemMessage(
                content="You are a financial assistant. Use the available tools to answer "
                "the user's question precisely. Call whichever tools are needed."
            ),
            HumanMessage(content=state["question"]),
        ]
        response: AIMessage = self.tool_model.invoke(messages)

        tool_calls: list[ToolCall] = []
        tool_results: list[dict] = []

        for call in getattr(response, "tool_calls", []) or []:
            tool = get_tools_by_name().get(call["name"])
            if tool is None:
                logger.warning("Model requested unknown tool: %s", call["name"])
                continue
            start = time.perf_counter()
            result = tool.execute(**call["args"])
            tool_calls.append(result)
            tool_results.append(
                {
                    "tool_name": tool.name,
                    "result": result.result if result.success else f"Error: {result.error}",
                }
            )
            logger.info(
                "Agent called tool '%s' args=%s success=%s (%.1fms)",
                tool.name,
                call["args"],
                result.success,
                (time.perf_counter() - start) * 1000,
            )

        return {**state, "tool_calls": tool_calls, "tool_results": tool_results}

    def _generate_node(self, state: AgentState) -> AgentState:
        sources = state.get("sources", [])
        tool_results = state.get("tool_results", [])

        prompt = self.prompt_builder.build(
            question=state["question"],
            sources=sources,
            conversation_history=state.get("conversation_history"),
            tool_results=tool_results,
        )
        answer = self.llm_client.complete(prompt)
        top_k = state.get("top_k") or self.settings.retrieval_top_k
        confidence = self._estimate_confidence(sources, tool_results, state.get("route", "direct"), top_k)

        return {**state, "answer": answer, "confidence": confidence}

    @staticmethod
    def _estimate_confidence(sources: list, tool_results: list, route: str, top_k: int = 5) -> float:
        if route == "tool" and tool_results:
            return 0.85 if all("Error" not in str(r.get("result", "")) for r in tool_results) else 0.3
        if sources:
            avg = sum(getattr(s, "score", 0.0) for s in sources) / len(sources)
            expected_min = max(1, int(top_k * 0.6))
            coverage = min(1.0, len(sources) / expected_min)
            return round(min(1.0, avg * coverage), 2)
        if route == "direct":
            return 0.6
        return 0.2

    # ------------------------------------------------------------------ #
    # Public entrypoint
    # ------------------------------------------------------------------ #
    def run(
        self,
        question: str,
        session_id: str,
        conversation_history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> ChatResponse:
        """Run the agent graph end-to-end for a single user turn.

        Args:
            question: The user's message.
            session_id: Conversation/session identifier.
            conversation_history: Prior turns for context.

        Returns:
            A fully populated ChatResponse.
        """
        start = time.perf_counter()
        initial_state: AgentState = {
            "question": question,
            "session_id": session_id,
            "conversation_history": conversation_history or [],
            "top_k": top_k,
        }
        final_state = self.graph.invoke(initial_state)
        latency_ms = (time.perf_counter() - start) * 1000

        return ChatResponse(
            session_id=session_id,
            answer=final_state.get("answer", ""),
            confidence=final_state.get("confidence", 0.0),
            sources=final_state.get("sources", []) or [],
            tool_calls=final_state.get("tool_calls", []) or [],
            route=final_state.get("route", "direct"),
            latency_ms=latency_ms,
        )
