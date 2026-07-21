"""Wraps FinAssist's MCP tools as LangChain StructuredTools.

This lets the LangGraph agent bind these tools directly to the LLM so the
model can decide when to call them (e.g. stock price, currency conversion)
during reasoning, while still going through the same validated,
logged execution path defined in :mod:`backend.mcp.base`.
"""

from langchain_core.tools import StructuredTool

from backend.mcp.tools.registry import get_all_tools
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_langchain_tools() -> list[StructuredTool]:
    """Convert every registered MCP tool into a LangChain StructuredTool.

    Returns:
        List of StructuredTool instances bindable to a LangChain chat model.
    """
    lc_tools = []
    for tool in get_all_tools():

        def make_func(bound_tool):
            def func(**kwargs):
                result = bound_tool.execute(**kwargs)
                if not result.success:
                    return f"Error calling {bound_tool.name}: {result.error}"
                return result.result

            return func

        lc_tools.append(
            StructuredTool.from_function(
                func=make_func(tool),
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_schema,
            )
        )
    logger.info("Built %s LangChain tools from MCP registry", len(lc_tools))
    return lc_tools
