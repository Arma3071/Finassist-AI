"""Model Context Protocol (MCP) server exposing FinAssist's financial tools.

Run standalone with:

    python -m backend.mcp.server

This starts a stdio MCP server that any MCP-compatible client (Claude
Desktop, other agents, etc.) can connect to and call these tools from,
independent of the in-process LangGraph agent used by the FastAPI backend.
"""

from mcp.server.fastmcp import FastMCP

from backend.mcp.tools.registry import get_all_tools
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

mcp_server = FastMCP("finassist-tools")


def _register_tools() -> None:
    """Register every BaseTool with the FastMCP server, preserving schema/validation."""
    for tool in get_all_tools():

        def make_handler(bound_tool):
            def handler(**kwargs):
                result = bound_tool.execute(**kwargs)
                if not result.success:
                    raise RuntimeError(result.error or "Tool execution failed")
                return result.result

            handler.__name__ = bound_tool.name
            handler.__doc__ = bound_tool.description
            return handler

        mcp_server.add_tool(
            make_handler(tool),
            name=tool.name,
            description=tool.description,
        )
        logger.info("Registered MCP tool: %s", tool.name)


_register_tools()

if __name__ == "__main__":
    mcp_server.run(transport="stdio")
