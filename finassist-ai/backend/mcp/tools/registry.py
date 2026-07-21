"""Central registry of all available MCP tools."""

from backend.mcp.base import BaseTool
from backend.mcp.tools.calculator import CalculatorTool
from backend.mcp.tools.currency_converter import CurrencyConverterTool
from backend.mcp.tools.datetime_tool import DateTimeTool
from backend.mcp.tools.financials import CompanyFinancialsTool
from backend.mcp.tools.news_search import NewsSearchTool
from backend.mcp.tools.stock_price import StockPriceTool

_TOOL_INSTANCES: list[BaseTool] | None = None


def get_all_tools() -> list[BaseTool]:
    """Instantiate and return every registered MCP tool.

    Returns:
        A list of BaseTool instances available to the agent and MCP server.
    """
    global _TOOL_INSTANCES
    if _TOOL_INSTANCES is None:
        _TOOL_INSTANCES = [
            StockPriceTool(),
            CompanyFinancialsTool(),
            CurrencyConverterTool(),
            NewsSearchTool(),
            CalculatorTool(),
            DateTimeTool(),
        ]
    return _TOOL_INSTANCES


def get_tools_by_name() -> dict[str, BaseTool]:
    """Return a mapping of tool name to tool instance, built lazily."""
    return {tool.name: tool for tool in get_all_tools()}
