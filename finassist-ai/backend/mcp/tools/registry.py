"""Central registry of all available MCP tools."""

from backend.mcp.base import BaseTool
from backend.mcp.tools.calculator import CalculatorTool
from backend.mcp.tools.currency_converter import CurrencyConverterTool
from backend.mcp.tools.datetime_tool import DateTimeTool
from backend.mcp.tools.financials import CompanyFinancialsTool
from backend.mcp.tools.news_search import NewsSearchTool
from backend.mcp.tools.stock_price import StockPriceTool


def get_all_tools() -> list[BaseTool]:
    """Instantiate and return every registered MCP tool.

    Returns:
        A list of BaseTool instances available to the agent and MCP server.
    """
    return [
        StockPriceTool(),
        CompanyFinancialsTool(),
        CurrencyConverterTool(),
        NewsSearchTool(),
        CalculatorTool(),
        DateTimeTool(),
    ]


TOOLS_BY_NAME: dict[str, BaseTool] = {tool.name: tool for tool in get_all_tools()}
