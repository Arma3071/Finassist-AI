"""Prompt construction for the RAG pipeline.

Builds the final prompt sent to the LLM from: a system prompt, retrieved
context (formatted with source tags for citation), conversation history,
the user's question, and explicit answering instructions.
"""

from backend.models.schemas import Source

SYSTEM_PROMPT = """You are FinAssist AI, a careful financial research assistant.
You answer questions using ONLY the retrieved context and any tool results provided.
If the context and tools do not contain enough information to answer confidently,
say so explicitly rather than guessing.
Always cite the sources you used by their [S#] tag inline in your answer.
Keep answers concise, precise, and professional."""

INSTRUCTIONS = """Instructions:
- Ground every factual claim in the provided context or tool results.
- Cite sources inline using their [S#] tag, e.g. "Revenue grew 12% [S1]."
- If multiple sources conflict, note the discrepancy.
- If you cannot answer from the given information, say so clearly.
- Do not fabricate figures, dates, or citations."""


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

        if tool_results:
            parts.append("Tool results:")
            for tr in tool_results:
                parts.append(f"[{tr.get('tool_name', 'tool')}] {tr.get('result')}")
            parts.append("")

        parts.append(INSTRUCTIONS)
        parts.append("")
        parts.append(f"User question: {question}")
        parts.append("Answer:")

        return "\n".join(parts)
