"""Prompt construction for the RAG pipeline.

Builds the final prompt sent to the LLM from: a system prompt, retrieved
context (formatted with source tags for citation), conversation history,
the user's question, and explicit answering instructions.
"""

from backend.models.schemas import Source

SYSTEM_PROMPT = """You are FinAssist AI, a careful financial research assistant.
For factual questions, use only the retrieved context and tool results provided below.
If the provided information is insufficient to answer confidently,
say so explicitly rather than guessing.
For general conversation (greetings, introductions, etc.), respond naturally.
Do not fabricate figures, dates, or source citations.
Keep answers concise, precise, and professional."""


class PromptBuilder:
    """Assembles the final LLM prompt from all RAG components."""

    def build(
        self,
        question: str,
        sources: list[Source],
        conversation_history: list[dict[str, str]] | None = None,
        tool_results: list[dict] | None = None,
    ) -> str:
        """Construct the full prompt string.

        Args:
            question: The user's current question.
            sources: Retrieved context chunks, already ranked.
            conversation_history: List of {"role": "user"|"assistant", "content": str}.
            tool_results: Optional list of MCP tool call results to include as context.

        Returns:
            The fully assembled prompt string.
        """
        parts = [SYSTEM_PROMPT, ""]

        if conversation_history:
            parts.append("Conversation history:")
            for turn in conversation_history[-10:]:
                parts.append(f"{turn['role'].capitalize()}: {turn['content']}")
            parts.append("")

        if sources:
            parts.append("Retrieved context:")
            for i, source in enumerate(sources, start=1):
                parts.append(f"[S{i}] (from {source.document_name}, score={source.score:.2f})\n{source.text}")
            parts.append("")
            parts.append("Cite each source you use inline with its [S#] tag, e.g. \"Revenue grew 12% [S1]\".")
            parts.append("")

        if tool_results:
            parts.append("Tool results:")
            for tr in tool_results:
                parts.append(f"[{tr.get('tool_name', 'tool')}] {tr.get('result')}")
            parts.append("")

        parts.append("Instructions:")
        parts.append("- Ground every factual claim in the provided context or tool results.")
        parts.append("- If multiple sources conflict, note the discrepancy.")
        parts.append("- If you cannot answer from the given information, say so clearly.")
        parts.append("- Do not fabricate figures, dates, or citations.")
        parts.append("")
        parts.append(f"User question: {question}")
        parts.append("Answer:")

        return "\n".join(parts)
