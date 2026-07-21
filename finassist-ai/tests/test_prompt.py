"""Unit tests for backend.rag.prompt.PromptBuilder."""

from backend.models.schemas import Source
from backend.rag.prompt import PromptBuilder


def test_prompt_includes_question_and_instructions():
    builder = PromptBuilder()
    prompt = builder.build(question="What was Q3 revenue?", sources=[])

    assert "What was Q3 revenue?" in prompt
    assert "Instructions:" in prompt


def test_prompt_includes_numbered_sources():
    builder = PromptBuilder()
    sources = [
        Source(chunk_id="1", document_name="report.md", text="Revenue was $482M.", score=0.9),
        Source(chunk_id="2", document_name="report.md", text="Margin was 61.2%.", score=0.8),
    ]
    prompt = builder.build(question="Summarize.", sources=sources)

    assert "[S1]" in prompt
    assert "[S2]" in prompt
    assert "Revenue was $482M." in prompt


def test_prompt_includes_conversation_history():
    builder = PromptBuilder()
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    prompt = builder.build(question="Continue.", sources=[], conversation_history=history)

    assert "User: Hi" in prompt
    assert "Assistant: Hello!" in prompt


def test_prompt_includes_tool_results():
    builder = PromptBuilder()
    prompt = builder.build(
        question="What's AAPL trading at?",
        sources=[],
        tool_results=[{"tool_name": "stock_price", "result": {"price": 210.5}}],
    )
    assert "stock_price" in prompt
